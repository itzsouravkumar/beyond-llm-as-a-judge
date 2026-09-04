from src.evaluators import SimulatedLLMJudge, SymbolicVerifier, ProgrammaticJudge

def test_simulated_llm_judge():
    judge = SimulatedLLMJudge()
    result = judge.evaluate("What is 2+2?", "4")
    assert "score" in result
    assert "verdict" in result
    assert result["verdict"] in ["pass", "fail"]

def test_symbolic_verifier():
    verifier = SymbolicVerifier()
    result = verifier.evaluate("Say hello", "hello world!")
    assert result["verdict"] == "pass"

def test_programmatic_judge():
    judge = ProgrammaticJudge()
    result = judge.evaluate("Write python add function", "def add():\n    return 0")
    assert result["verdict"] == "pass"
