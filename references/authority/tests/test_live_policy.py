from minagi.live import exchange_text


def test_user_only_live_learning_text():
    text = exchange_text("new fact", "model guess", include_bot=False)
    assert "new fact" in text
    assert "model guess" not in text
    assert "<bot>" not in text


def test_legacy_full_exchange_still_available():
    text = exchange_text("question", "answer", include_bot=True)
    assert "question" in text and "answer" in text
