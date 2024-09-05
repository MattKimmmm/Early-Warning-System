import sentence_transformers.evaluation as evaluation
import sentence_transformers.util as util
import logging, torch, time, datetime

class TripletAccuracyEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, eval_dataset, batch_size=16, device='cuda'):
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.device = device

    def __call__(self, model):
        since = time.time()

        model = model.to(self.device)
        total_correct = 0
        total_comparisons = 0

        # Process dataset in batches directly
        for i in range(0, len(self.eval_dataset['sentence']), self.batch_size):
            batch_sentences = self.eval_dataset['sentence'][i:i+self.batch_size]
            batch_labels = self.eval_dataset['label'][i:i+self.batch_size]

            embeddings = model.encode(batch_sentences, batch_size=self.batch_size, convert_to_tensor=True, device=self.device)

            if i % 10 == 0:
                logging.info(f"Processing batch {i // self.batch_size + 1}/{len(self.eval_dataset['sentence']) // self.batch_size}")
                logging.info(f"Current Memory Usage: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB allocated, {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB reserved")

            # Compare within this batch
            for j, anchor_embedding in enumerate(embeddings):
                anchor_label = batch_labels[j]
                
                # Find positive and negative pairs within the batch
                for k, compare_embedding in enumerate(embeddings):
                    if j == k:
                        continue
                    compare_label = batch_labels[k]
                    
                    pos_dist = util.pytorch_cos_sim(anchor_embedding, compare_embedding) if anchor_label == compare_label else None
                    neg_dist = util.pytorch_cos_sim(anchor_embedding, compare_embedding) if anchor_label != compare_label else None
                    
                    if pos_dist is not None and neg_dist is not None and pos_dist > neg_dist:
                        total_correct += 1
                    total_comparisons += 1
            
            if i % 10 == 0:
                print(f"Batch {i // self.batch_size + 1} done in {datetime.timedelta(seconds=(time.time() - since))}.")

        accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0
        print(f"Evaluation accuracy: {accuracy:.4f}")

        return accuracy