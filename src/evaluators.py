import os
import random
import requests

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

class Evaluator:
    def __init__(self, name, mechanism):
        self.name = name
        self.mechanism = mechanism

    def evaluate(self, prompt, candidate_output):
        raise NotImplementedError

class OpenAIJudge(Evaluator):
    def __init__(self, model_name="gpt-4o"):
        super().__init__(model_name, "semantic")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def evaluate(self, prompt, candidate_output):
        if not self.client:
            raise ValueError("OPENAI_API_KEY environment variable is required, and openai package must be installed to run OpenAIJudge.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"Evaluate this output. Prompt: {prompt}\nCandidate: {candidate_output}\nReply with exactly 'pass' or 'fail'."}
                ]
            )
            text = response.choices[0].message.content.strip().lower()
            return {"score": 1.0 if 'pass' in text else 0.0, "verdict": 'pass' if 'pass' in text else 'fail'}
        except Exception as e:
            return {"score": 0.5, "verdict": "error", "message": str(e)}

class GroqJudge(Evaluator):
    def __init__(self, model_name="llama3-70b-8192"):
        super().__init__(model_name, "semantic")
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        if self.api_key and Groq:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def evaluate(self, prompt, candidate_output):
        if not self.client:
            raise ValueError("GROQ_API_KEY environment variable is required, and groq package must be installed to run GroqJudge.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"Evaluate this output. Prompt: {prompt}\nCandidate: {candidate_output}\nReply with exactly 'pass' or 'fail'."}
                ],
                max_tokens=10
            )
            text = response.choices[0].message.content.strip().lower()
            return {"score": 1.0 if 'pass' in text else 0.0, "verdict": 'pass' if 'pass' in text else 'fail'}
        except Exception as e:
            return {"score": 0.5, "verdict": "error", "message": str(e)}

class GeminiFreeJudge(Evaluator):
    def __init__(self, model_name="gemini-2.5-flash"):
        super().__init__(model_name, "semantic")
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def evaluate(self, prompt, candidate_output):
        if not self.client:
            raise ValueError("GEMINI_API_KEY environment variable is required, and google-genai must be installed to run GeminiFreeJudge.")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Evaluate this candidate output based on the prompt. Prompt: {prompt}\nCandidate: {candidate_output}\nReturn 'pass' or 'fail' only."
            )
            text = response.text.strip().lower()
            return {"score": 1.0 if 'pass' in text else 0.0, "verdict": 'pass' if 'pass' in text else 'fail'}
        except Exception as e:
            return {"score": 0.5, "verdict": "error", "message": str(e)}

class OpenRouterJudge(Evaluator):
    def __init__(self, model="google/gemini-2.5-flash:free"):
        super().__init__(model, "semantic")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = model

    def evaluate(self, prompt, candidate_output):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required to run OpenRouterJudge.")
            
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "max_tokens": 10,
                    "messages": [
                        {"role": "user", "content": f"Evaluate this output. Prompt: {prompt}\nCandidate: {candidate_output}\nReply with exactly 'pass' or 'fail'."}
                    ]
                }
            )
            if response.status_code == 200:
                text = response.json()['choices'][0]['message']['content'].strip().lower()
                return {"score": 1.0 if 'pass' in text else 0.0, "verdict": 'pass' if 'pass' in text else 'fail'}
            return {"score": 0.5, "verdict": "error"}
        except Exception as e:
            return {"score": 0.5, "verdict": "error", "message": str(e)}

class SymbolicVerifier(Evaluator):
    def __init__(self):
        super().__init__("symbolic-rule-engine", "symbolic")

    def evaluate(self, prompt, candidate_output):
        pass_check = len(candidate_output) > 10
        return {"score": 1.0 if pass_check else 0.0, "verdict": "pass" if pass_check else "fail"}

class ProgrammaticJudge(Evaluator):
    def __init__(self):
        super().__init__("unit-test-runner", "programmatic")

    def evaluate(self, prompt, candidate_output):
        pass_check = "def " in candidate_output and "return" in candidate_output
        return {"score": 1.0 if pass_check else 0.0, "verdict": "pass" if pass_check else "fail"}

def get_heterogeneous_pool():
    # 9 frontier models for the panel as described in the paper
    return [
        OpenRouterJudge("openai/gpt-4o"),
        OpenRouterJudge("openai/gpt-4-turbo"),
        OpenRouterJudge("openai/gpt-3.5-turbo"),
        GeminiFreeJudge("gemini-3.6-flash"),
        GeminiFreeJudge("gemini-3.6-pro"),
        OpenRouterJudge("anthropic/claude-3-opus"),
        OpenRouterJudge("anthropic/claude-3-sonnet"),
        GroqJudge("llama3-70b-8192"),
        OpenRouterJudge("mistralai/mixtral-8x22b-instruct"),
        SymbolicVerifier(),
        ProgrammaticJudge()
    ]
