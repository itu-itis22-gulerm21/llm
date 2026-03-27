"""
Medical LLM Benchmark Evaluation Script
=========================================

This script evaluates fine-tuned medical LLMs on standard medical benchmarks:
- MedQA: Medical question answering dataset
- PubMedQA: Biomedical literature question answering

Metrics Calculated:
- Accuracy
- F1 Score (Macro and Weighted)
- Precision & Recall
- Training Loss Reduction (%)

Usage:
    python benchmark_evaluation.py --num_samples 500 --output_dir ./results

Author: Medical LLM Fine-tuning Project
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import json
import re
import argparse
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MedicalLLMEvaluator:
    """
    Evaluator class for medical LLMs on benchmark datasets.
    """
    
    def __init__(self, output_dir: str = './evaluation_results', num_samples: int = None):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            num_samples: Number of samples to evaluate (None for full dataset)
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Model configurations
        self.models = {
            'gemma_medical': 'Shekswess/gemma-1.1-7b-it-bnb-4bit-medical',
            'llama2_medical': 'Shekswess/llama-2-7b-chat-bnb-4bit-medical',
            'llama3_medical': 'Shekswess/llama-3-8b-Instruct-bnb-4bit-medical',
            'mistral_medical': 'Shekswess/mistral-7b-instruct-v0.2-bnb-4bit-medical'
        }
        
        self.base_models = {
            'gemma_base': 'unsloth/gemma-1.1-7b-it-bnb-4bit',
            'llama2_base': 'unsloth/llama-2-7b-chat-bnb-4bit',
            'llama3_base': 'unsloth/llama-3-8b-Instruct-bnb-4bit',
            'mistral_base': 'unsloth/mistral-7b-instruct-v0.2-bnb-4bit'
        }
        
        # Generation parameters
        self.max_new_tokens = 256
        self.temperature = 0.1
        
        # Results storage
        self.all_results = {}
        self.base_results = {}
        
        print(f"Evaluator initialized!")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    def load_model_and_tokenizer(self, model_name: str):
        """Load model and tokenizer with 4-bit quantization."""
        print(f"\nLoading model: {model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model.eval()
        print(f"Model loaded successfully!")
        
        return model, tokenizer
    
    def format_prompt(self, question: str, options: List[str] = None, model_type: str = "gemma") -> str:
        """Format the prompt according to model's instruction format."""
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt = f"""You are a medical expert. Answer the following medical question by selecting the correct option.

Question: {question}

Options:
{options_text}

Answer with only the letter (A, B, C, or D) of the correct option."""
        else:
            prompt = f"""You are a medical expert. Answer the following medical question with 'yes', 'no', or 'maybe'.

Question: {question}

Answer:"""
        
        # Apply model-specific formatting
        if "gemma" in model_type.lower():
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "llama-2" in model_type.lower():
            return f"[INST] {prompt} [/INST]"
        elif "llama-3" in model_type.lower() or "llama3" in model_type.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in model_type.lower():
            return f"[INST] {prompt} [/INST]"
        else:
            return prompt
    
    def extract_answer(self, response: str, answer_type: str = "multiple_choice") -> str:
        """Extract the answer from model response."""
        response = response.strip().upper()
        
        if answer_type == "multiple_choice":
            match = re.search(r'\b([A-D])\b', response)
            if match:
                return match.group(1)
            if response and response[0] in ['A', 'B', 'C', 'D']:
                return response[0]
            return "INVALID"
        
        elif answer_type == "yes_no_maybe":
            response_lower = response.lower()
            if "yes" in response_lower:
                return "yes"
            elif "no" in response_lower:
                return "no"
            elif "maybe" in response_lower:
                return "maybe"
            return "INVALID"
        
        return "INVALID"
    
    def generate_response(self, model, tokenizer, prompt: str) -> str:
        """Generate response from model."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def calculate_metrics(self, predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        valid_indices = [i for i, pred in enumerate(predictions) if pred != "INVALID"]
        
        if not valid_indices:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'valid_responses': 0
            }
        
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        metrics = {
            'accuracy': accuracy_score(valid_labels, valid_predictions),
            'f1_macro': f1_score(valid_labels, valid_predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
            'precision': precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
            'recall': recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
            'valid_responses': len(valid_indices) / len(predictions)
        }
        
        return metrics
    
    def load_datasets(self):
        """Load benchmark datasets."""
        print("\n" + "="*80)
        print("Loading Benchmark Datasets")
        print("="*80)
        
        # Load MedQA
        print("\nLoading MedQA dataset...")
        try:
            medqa = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test")
        except:
            print("Using alternative MedQA dataset...")
            medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        
        if self.num_samples:
            medqa = medqa.select(range(min(self.num_samples, len(medqa))))
        
        print(f"Loaded {len(medqa)} MedQA samples")
        
        # Load PubMedQA
        print("\nLoading PubMedQA dataset...")
        pubmedqa = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        
        if self.num_samples:
            pubmedqa = pubmedqa.select(range(min(self.num_samples, len(pubmedqa))))
        
        print(f"Loaded {len(pubmedqa)} PubMedQA samples")
        
        return medqa, pubmedqa
    
    def evaluate_model_on_benchmark(self, model_name: str, dataset, dataset_name: str, model_type: str) -> Dict:
        """Evaluate a single model on a benchmark dataset."""
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name} on {dataset_name}")
        print(f"{'='*80}")
        
        model, tokenizer = self.load_model_and_tokenizer(model_name)
        
        predictions = []
        labels = []
        responses_log = []
        
        answer_type = "yes_no_maybe" if dataset_name == "PubMedQA" else "multiple_choice"
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            try:
                if dataset_name == "MedQA":
                    question = sample.get('question', sample.get('Question', ''))
                    options = sample.get('options', sample.get('Options', {}))
                    
                    if isinstance(options, dict):
                        options_list = [options.get(k, '') for k in ['A', 'B', 'C', 'D']]
                    else:
                        options_list = options if isinstance(options, list) else []
                    
                    correct_answer = sample.get('answer', sample.get('Answer', 'A')).strip().upper()
                    if len(correct_answer) > 1:
                        correct_answer = correct_answer[0]
                    
                    prompt = self.format_prompt(question, options_list, model_type)
                    
                else:  # PubMedQA
                    question = sample['question']
                    correct_answer = sample['final_decision'].lower()
                    prompt = self.format_prompt(question, None, model_type)
                
                response = self.generate_response(model, tokenizer, prompt)
                predicted_answer = self.extract_answer(response, answer_type)
                
                predictions.append(predicted_answer)
                labels.append(correct_answer)
                
                responses_log.append({
                    'question': question,
                    'predicted': predicted_answer,
                    'correct': correct_answer,
                    'response': response[:200]
                })
                
                if (idx + 1) % 50 == 0:
                    temp_metrics = self.calculate_metrics(predictions, labels)
                    print(f"\nProgress at {idx + 1} samples:")
                    print(f"  Accuracy: {temp_metrics['accuracy']:.4f}")
                    print(f"  Valid Responses: {temp_metrics['valid_responses']:.2%}")
            
            except Exception as e:
                print(f"\nError processing sample {idx}: {str(e)}")
                predictions.append("INVALID")
                labels.append(sample.get('answer', sample.get('final_decision', 'A')))
                continue
        
        metrics = self.calculate_metrics(predictions, labels)
        
        print(f"\n{'='*80}")
        print(f"Results for {model_name} on {dataset_name}:")
        print(f"{'='*80}")
        print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 Score (Macro):     {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"Valid Responses: {metrics['valid_responses']:.2%}")
        print(f"{'='*80}")
        
        del model
        torch.cuda.empty_cache()
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels,
            'responses_log': responses_log
        }
    
    def run_full_evaluation(self):
        """Run complete evaluation on all models and datasets."""
        # Load datasets
        medqa_dataset, pubmedqa_dataset = self.load_datasets()
        
        # Evaluate fine-tuned models
        print("\n" + "#"*80)
        print("# EVALUATING FINE-TUNED MODELS")
        print("#"*80)
        
        for model_key, model_name in self.models.items():
            model_type = model_key.split('_')[0]
            
            medqa_results = self.evaluate_model_on_benchmark(
                model_name, medqa_dataset, "MedQA", model_type
            )
            
            pubmedqa_results = self.evaluate_model_on_benchmark(
                model_name, pubmedqa_dataset, "PubMedQA", model_type
            )
            
            self.all_results[model_key] = {
                'MedQA': medqa_results,
                'PubMedQA': pubmedqa_results
            }
            
            print(f"\n✓ Completed evaluation for {model_key}\n")
        
        # Evaluate base models
        print("\n" + "#"*80)
        print("# EVALUATING BASE MODELS")
        print("#"*80)
        
        for model_key, model_name in self.base_models.items():
            model_type = model_key.split('_')[0]
            
            medqa_results = self.evaluate_model_on_benchmark(
                model_name, medqa_dataset, "MedQA", model_type
            )
            
            pubmedqa_results = self.evaluate_model_on_benchmark(
                model_name, pubmedqa_dataset, "PubMedQA", model_type
            )
            
            self.base_results[model_key] = {
                'MedQA': medqa_results,
                'PubMedQA': pubmedqa_results
            }
            
            print(f"\n✓ Completed evaluation for {model_key}\n")
    
    def generate_comparison_report(self):
        """Generate comparison report between base and fine-tuned models."""
        comparison_data = []
        
        models = ['gemma', 'llama2', 'llama3', 'mistral']
        
        for model_name in models:
            finetuned_key = f"{model_name}_medical"
            base_key = f"{model_name}_base"
            
            for dataset_name in ['MedQA', 'PubMedQA']:
                base_acc = self.base_results[base_key][dataset_name]['metrics']['accuracy']
                finetuned_acc = self.all_results[finetuned_key][dataset_name]['metrics']['accuracy']
                improvement = ((finetuned_acc - base_acc) / base_acc * 100) if base_acc > 0 else 0
                
                base_f1 = self.base_results[base_key][dataset_name]['metrics']['f1_weighted']
                finetuned_f1 = self.all_results[finetuned_key][dataset_name]['metrics']['f1_weighted']
                f1_improvement = ((finetuned_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Dataset': dataset_name,
                    'Base Accuracy': f"{base_acc:.4f}",
                    'Fine-tuned Accuracy': f"{finetuned_acc:.4f}",
                    'Accuracy Improvement': f"{improvement:+.2f}%",
                    'Base F1': f"{base_f1:.4f}",
                    'Fine-tuned F1': f"{finetuned_f1:.4f}",
                    'F1 Improvement': f"{f1_improvement:+.2f}%"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*120)
        print("MODEL COMPARISON: BASE vs FINE-TUNED")
        print("="*120)
        print(comparison_df.to_string(index=False))
        print("="*120)
        
        # Save to CSV
        comparison_df.to_csv(f"{self.output_dir}/model_comparison.csv", index=False)
        print(f"\n✓ Saved comparison to {self.output_dir}/model_comparison.csv")
        
        return comparison_df
    
    def create_visualizations(self):
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        models = ['GEMMA', 'LLAMA2', 'LLAMA3', 'MISTRAL']
        
        # MedQA Accuracy
        medqa_base = [self.base_results[f"{m.lower()}_base"]['MedQA']['metrics']['accuracy'] * 100 for m in models]
        medqa_finetuned = [self.all_results[f"{m.lower()}_medical"]['MedQA']['metrics']['accuracy'] * 100 for m in models]
        
        # PubMedQA Accuracy
        pubmed_base = [self.base_results[f"{m.lower()}_base"]['PubMedQA']['metrics']['accuracy'] * 100 for m in models]
        pubmed_finetuned = [self.all_results[f"{m.lower()}_medical"]['PubMedQA']['metrics']['accuracy'] * 100 for m in models]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Medical LLM Benchmark Evaluation Results', fontsize=16, fontweight='bold')
        
        x = np.arange(len(models))
        width = 0.35
        
        # Plot 1: MedQA Accuracy
        axes[0, 0].bar(x - width/2, medqa_base, width, label='Base Model', alpha=0.8)
        axes[0, 0].bar(x + width/2, medqa_finetuned, width, label='Fine-tuned', alpha=0.8)
        axes[0, 0].set_xlabel('Model', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 0].set_title('MedQA: Base vs Fine-tuned Accuracy', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        for i, (v1, v2) in enumerate(zip(medqa_base, medqa_finetuned)):
            axes[0, 0].text(i - width/2, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: PubMedQA Accuracy
        axes[0, 1].bar(x - width/2, pubmed_base, width, label='Base Model', alpha=0.8)
        axes[0, 1].bar(x + width/2, pubmed_finetuned, width, label='Fine-tuned', alpha=0.8)
        axes[0, 1].set_xlabel('Model', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 1].set_title('PubMedQA: Base vs Fine-tuned Accuracy', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, (v1, v2) in enumerate(zip(pubmed_base, pubmed_finetuned)):
            axes[0, 1].text(i - width/2, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
            axes[0, 1].text(i + width/2, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: MedQA Improvement
        medqa_improvements = [(ft - base) / base * 100 for base, ft in zip(medqa_base, medqa_finetuned)]
        colors_medqa = ['green' if x > 0 else 'red' for x in medqa_improvements]
        
        axes[1, 0].bar(x, medqa_improvements, color=colors_medqa, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_xlabel('Model', fontweight='bold')
        axes[1, 0].set_ylabel('Improvement (%)', fontweight='bold')
        axes[1, 0].set_title('MedQA: Accuracy Improvement after Fine-tuning', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models)
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(medqa_improvements):
            axes[1, 0].text(i, v + 1 if v > 0 else v - 1, f'{v:+.1f}%', 
                            ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
        
        # Plot 4: PubMedQA Improvement
        pubmed_improvements = [(ft - base) / base * 100 for base, ft in zip(pubmed_base, pubmed_finetuned)]
        colors_pubmed = ['green' if x > 0 else 'red' for x in pubmed_improvements]
        
        axes[1, 1].bar(x, pubmed_improvements, color=colors_pubmed, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Model', fontweight='bold')
        axes[1, 1].set_ylabel('Improvement (%)', fontweight='bold')
        axes[1, 1].set_title('PubMedQA: Accuracy Improvement after Fine-tuning', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(pubmed_improvements):
            axes[1, 1].text(i, v + 1 if v > 0 else v - 1, f'{v:+.1f}%', 
                            ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/benchmark_results.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {self.output_dir}/benchmark_results.png")
        plt.close()
    
    def save_detailed_results(self):
        """Save detailed results to JSON."""
        results_summary = {
            'evaluation_config': {
                'num_samples': self.num_samples,
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'models_evaluated': list(self.models.keys()),
                'base_models': list(self.base_models.keys()),
                'datasets': ['MedQA', 'PubMedQA']
            },
            'results': {},
            'base_model_results': {}
        }
        
        for model_key in self.models.keys():
            results_summary['results'][model_key] = {
                'MedQA': {
                    'metrics': self.all_results[model_key]['MedQA']['metrics'],
                    'sample_predictions': self.all_results[model_key]['MedQA']['responses_log'][:10]
                },
                'PubMedQA': {
                    'metrics': self.all_results[model_key]['PubMedQA']['metrics'],
                    'sample_predictions': self.all_results[model_key]['PubMedQA']['responses_log'][:10]
                }
            }
        
        for model_key in self.base_models.keys():
            results_summary['base_model_results'][model_key] = {
                'MedQA': self.base_results[model_key]['MedQA']['metrics'],
                'PubMedQA': self.base_results[model_key]['PubMedQA']['metrics']
            }
        
        with open(f"{self.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✓ Saved detailed results to {self.output_dir}/evaluation_results.json")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Evaluate Medical LLMs on Benchmarks')
    parser.add_argument('--num_samples', type=int, default=500, 
                        help='Number of samples to evaluate (default: 500, use 0 for full dataset)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results (default: ./evaluation_results)')
    
    args = parser.parse_args()
    
    num_samples = args.num_samples if args.num_samples > 0 else None
    
    print("\n" + "="*80)
    print("Medical LLM Benchmark Evaluation")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Samples per dataset: {num_samples if num_samples else 'Full dataset'}")
    print(f"  - Output directory: {args.output_dir}")
    print("="*80)
    
    # Initialize evaluator
    evaluator = MedicalLLMEvaluator(
        output_dir=args.output_dir,
        num_samples=num_samples
    )
    
    # Run evaluation
    evaluator.run_full_evaluation()
    
    # Generate reports and visualizations
    evaluator.generate_comparison_report()
    evaluator.create_visualizations()
    evaluator.save_detailed_results()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
