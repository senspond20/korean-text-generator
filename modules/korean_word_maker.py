from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

class KoreanWordMaker():

    def __init__(self):
        model = GPT2LMHeadModel.from_pretrained(
            "skt/kogpt2-base-v2"
        )
        model.eval()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "skt/kogpt2-base-v2",
            eos_token="</s>"
        )
        self.model = model
        self.tokenizer = tokenizer

    def make_text(self, max_length, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        """그리디 서치"""
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,          # False이면 컨텍스트가 동일하면 결과값이 항상 같으
                min_length=50,
                max_length=max_length,
                no_repeat_ngram_size=3,  # 토큰이 3개 이상 반복될 경우 3번째 토큰 확률을 0으로
                repetition_penalty=1.5,  # 반복 패널티
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                top_k=50,
                temperature=1           # 창의성y

            )

        """토큰 익덱스를 가지고 문장생성"""
        return self.tokenizer.decode([el.item() for el in generated_ids[0]])
