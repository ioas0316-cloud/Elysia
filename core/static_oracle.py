import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class StaticOracle:
    """
    정적 오라클 (The Frozen CD) - MRI Edition
    학습이 완료된 소형 LLM(KoGPT-2 등)을 로드하여, 텍스트가 아닌
    내부 전자기장(Hidden States Tensor)을 스캔하여 반환하는 모듈.
    """
    def __init__(self, model_name="skt/kogpt2-base-v2"):
        self.model_name = model_name
        print(f"💿 [Static Oracle] 얼어붙은 CD(정적 모델) 로드 중: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # output_hidden_states=True 옵션으로 내부 뇌파 관측 허용
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True)
        
        # 완전한 정적 구조를 위해 평가 모드(eval) 진입
        self.model.eval()
        print("💿 [Static Oracle] 로드 완료.")
        
    def mri_scan(self, prompt: str):
        """
        프롬프트를 주입하고 마지막 레이어의 가장 마지막 토큰 Hidden States 텐서를 추출 (단일 MRI 스캔)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1][0, -1, :]
        return last_hidden_state

    def get_embedding_matrix(self) -> torch.Tensor:
        """오라클의 전체 어휘장(Vocabulary) 임베딩 매트릭스를 반환합니다. (통상 51200 x 768)"""
        # GPT-2 기반 아키텍처의 경우:
        return self.model.transformer.wte.weight.detach()
        
    def generate_and_scan(self, prompt: str, max_length: int = 15):
        """
        프롬프트에 대한 원본 텍스트를 생성하고, 생성된 각 토큰의 마지막 레이어 hidden state를 스캔하여 반환합니다.
        반환값: (생성된 전체 텍스트, 생성된 토큰들의 hidden states 리스트)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False # 결정론적 원본 추출을 위해 Greedy Search
            )
            
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # outputs.hidden_states는 튜플의 튜플 구조. 
        # (생성된 토큰 수, (레이어 수, 배치, 1, 768))
        # 생성된 각 토큰의 마지막 레이어 hidden state만 추출합니다.
        hidden_states_list = []
        
        # 첫 번째 입력(프롬프트)에 대한 hidden_states는 시퀀스 전체를 포함하므로 마지막 토큰만 사용
        first_step_hidden = outputs.hidden_states[0][-1][0, -1, :]
        hidden_states_list.append(first_step_hidden)
        
        # 이후 생성된 각 토큰의 hidden state
        for step_hidden in outputs.hidden_states[1:]:
            token_hidden = step_hidden[-1][0, 0, :]
            hidden_states_list.append(token_hidden)
            
        return generated_text, hidden_states_list
