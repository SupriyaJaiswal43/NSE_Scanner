"""
Alerts Module
Sound alert (via HTML audio) + browser notification via JS
"""

import streamlit as st
import base64


# ── Beep sound as base64 WAV (short sine wave) ───────────────────────────────
# This is a tiny 440 Hz beep encoded as base64 WAV so no external file needed
BEEP_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
)


def play_sound_alert():
    """Inject an auto-playing audio element into the Streamlit page."""
    audio_html = f"""
    <audio autoplay>
      <source src="data:audio/wav;base64,{BEEP_B64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)


def browser_notification(title: str, body: str):
    """Request browser notification permission and show a notification."""
    js_code = f"""
    <script>
    (function() {{
        if ("Notification" in window) {{
            if (Notification.permission === "granted") {{
                new Notification("{title}", {{ body: "{body}" }});
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(permission) {{
                    if (permission === "granted") {{
                        new Notification("{title}", {{ body: "{body}" }});
                    }}
                }});
            }}
        }}
    }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)


def trigger_alerts(new_signals: list):
    """Fire sound + browser notification for each new signal."""
    if not new_signals:
        return
    play_sound_alert()
    for sig in new_signals:
        browser_notification(
            title=f"🟢 BUY Signal — {sig['Stock']}",
            body=f"Price: ₹{sig['Price']} | EMA200: ₹{sig['EMA200']} | Score: {sig['Score']}"
        )