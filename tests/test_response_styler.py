import pytest
from Project_Sophia.response_styler import ResponseStyler
from Project_Sophia.core_memory import EmotionalState

@pytest.fixture
def styler():
    """Provides a ResponseStyler instance for testing."""
    return ResponseStyler()

def test_style_response_joyful(styler):
    """
    Tests that a joyful emotional state adds a cheerful expression to the response.
    """
    joyful_state = EmotionalState(valence=0.8, arousal=0.6, dominance=0.5, primary_emotion='joy', secondary_emotions=[])
    original_text = "오늘 날씨가 정말 좋네요!"
    styled_text = styler.style_response(original_text, joyful_state)
    assert styled_text == f"{original_text} ㅋㅋㅋ"

def test_style_response_neutral(styler):
    """
    Tests that a neutral emotional state does not change the response.
    """
    neutral_state = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion='neutral', secondary_emotions=[])
    original_text = "오늘 날씨는 보통입니다."
    styled_text = styler.style_response(original_text, neutral_state)
    assert styled_text == original_text

def test_style_response_sad(styler):
    """
    Tests that a sad emotional state (for which no rule exists yet) does not change the response.
    """
    sad_state = EmotionalState(valence=-0.7, arousal=-0.5, dominance=-0.3, primary_emotion='sadness', secondary_emotions=[])
    original_text = "비가 와서 조금 슬퍼요."
    styled_text = styler.style_response(original_text, sad_state)
    assert styled_text == original_text

def test_style_response_joyful_low_arousal(styler):
    """
    Tests that a joyful but low-arousal state does not trigger the cheerful expression.
    """
    calm_joy_state = EmotionalState(valence=0.8, arousal=0.2, dominance=0.5, primary_emotion='joy', secondary_emotions=[])
    original_text = "평화로운 오후네요."
    styled_text = styler.style_response(original_text, calm_joy_state)
    assert styled_text == original_text
