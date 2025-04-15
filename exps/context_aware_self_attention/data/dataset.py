import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any

class DADataset(Dataset):
    """
    A Dataset class for Dialog Act classification on the Switchboard dataset.
    The class expects a tokenizer (e.g., from the Hugging Face Transformers library),
    as well as a data structure containing text and label fields.
    """
    def __init__(
        self,
        tokenizer: Any,
        data: Dict[str, List[str]],
        text_field: str = "clean_text",
        label_field: str = "act_label_1",
        max_len: int = 512
    ) -> None:
        """
        Args:
            tokenizer (Any): A tokenizer instance that has an `encode_plus` method (e.g., a Hugging Face tokenizer).
            data (dict): A dictionary containing the data. Must have `text_field` and `label_field` keys.
            text_field (str): The field name in `data` that contains the text. Defaults to "clean_text".
            label_field (str): The field name in `data` that contains the labels. Defaults to "act_label_1".
            max_len (int): The maximum sequence length to pad or truncate to. Defaults to 512.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = list(data[text_field])
        self.acts = list(data[label_field])
        self.max_len = max_len

        self.label_dict = self._build_label_dict(self.acts)

    @staticmethod
    def _build_label_dict(acts: List[str]) -> Dict[str, int]:
        """
        Construct a dictionary mapping each unique label to an integer index.
        
        Args:
            acts (List[str]): A list of labels for each sample in the dataset.
            
        Returns:
            Dict[str, int]: A dictionary from label string -> integer index.
        """
        unique_labels = sorted(set(acts))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            A dictionary containing:
                - text (str): Original text from the dataset
                - input_ids (torch.LongTensor): Tokenized text indices
                - attention_mask (torch.LongTensor): Attention mask for the tokenized text
                - seq_len (int): The number of tokens (before padding)
                - act (str): The original label string
                - label (torch.LongTensor): The integer-encoded label
        """
        text = self.texts[idx]
        act = self.acts[idx]
        label = self.label_dict[act]

        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )

        seq_len = len(self.tokenizer.tokenize(text))

        return {
            "text": text,
            "input_ids": input_encoding["input_ids"].squeeze(0),   
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "seq_len": seq_len,
            "act": act,
            "label": torch.tensor(label, dtype=torch.long),
        }


# if __name__ == '__main__':
#     class MockTokenizer:
#         def encode_plus(
#             self, 
#             text, 
#             truncation=True, 
#             max_length=512, 
#             return_tensors="pt", 
#             return_attention_mask=True, 
#             padding="max_length"
#         ):
#             token_ids = list(range(1, max_length+1))
#             attention_mask = [1] * max_length
#             return {
#                 "input_ids": torch.tensor([token_ids], dtype=torch.long),
#                 "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
#             }

#         def tokenize(self, text):
#             return text.split()

#     data = {
#         "clean_text": [
#             "Hello how are you",
#             "I am fine",
#             "What about you",
#             "I am great too thanks for asking"
#         ],
#         "act_label_1": [
#             "Greeting",
#             "Statement",
#             "Question",
#             "Statement"
#         ]
#     }

#     tokenizer = MockTokenizer()
#     dataset = DADataset(tokenizer, data, text_field="clean_text", label_field="act_label_1", max_len=10)

#     print("Label dictionary:", dataset.label_dict)
#     print("Dataset length:", len(dataset))

#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

#     for batch_idx, batch in enumerate(dataloader):
#         print(f"\n--- Batch {batch_idx} ---")
#         print("Text:", batch["text"])
#         print("Input IDs:", batch["input_ids"])
#         print("Attention Mask:", batch["attention_mask"])
#         print("Sequence Lengths:", batch["seq_len"])
#         print("Acts:", batch["act"])
#         print("Labels:", batch["label"])

