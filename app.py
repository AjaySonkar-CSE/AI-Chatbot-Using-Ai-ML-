from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import json
import threading
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import queue

app = Flask(__name__, template_folder="templates")

# ---------------------------
# Configuration / Load keys
# ---------------------------
load_dotenv()  # Load .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# Load portfolio JSON
# ---------------------------
DATA_PATH = Path("data.json")
if not DATA_PATH.exists():
    raise RuntimeError("data.json not found. Put your portfolio JSON in data.json")

with DATA_PATH.open("r", encoding="utf-8") as f:
    portfolio = json.load(f)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------
# Background History Manager
# ---------------------------
HISTORY_PATH = Path("history.json")
HISTORY_MAX_ITEMS = 200

class HistoryManager:
    def __init__(self):
        self.history = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="history")
        self.write_queue = queue.Queue()
        self.recent_cache = []
        self.cache_size = 10
        
        # Load history in background
        self.executor.submit(self._load_history_background)
        
        # Start background writer
        self.executor.submit(self._background_writer)
    
    def _load_history_background(self):
        """Load history in background"""
        try:
            if not HISTORY_PATH.exists():
                print("📁 Creating new history.json")
                with HISTORY_PATH.open("w", encoding="utf-8") as fh:
                    json.dump([], fh, ensure_ascii=False, indent=2)
                return
            
            with HISTORY_PATH.open("r", encoding="utf-8") as fh:
                if HISTORY_PATH.stat().st_size == 0:
                    return
                
                data = json.load(fh)
                if isinstance(data, list):
                    with self.lock:
                        self.history = data
                        # Update recent cache
                        self.recent_cache = self.history[-self.cache_size:] if self.history else []
                    print(f"📚 History loaded: {len(data)} entries")
                else:
                    print("❌ Invalid history format, starting fresh")
                    
        except Exception as e:
            print(f"❌ Error loading history: {e}")
    
    def _background_writer(self):
        """Background thread to handle file writes"""
        while True:
            try:
                # Wait for write request
                write_data = self.write_queue.get(timeout=1)
                if write_data is None:  # Shutdown signal
                    break
                
                # Write to file
                with HISTORY_PATH.open("w", encoding="utf-8") as fh:
                    json.dump(write_data, fh, ensure_ascii=False, indent=2)
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Background write error: {e}")
    
    def get_recent_context(self, last_n=5):
        """Get recent context quickly from cache or memory"""
        with self.lock:
            if self.recent_cache:
                recent = self.recent_cache[-last_n:] if len(self.recent_cache) >= last_n else self.recent_cache
            else:
                # Fallback to main history if cache not ready
                recent = self.history[-last_n:] if self.history else []
            
        return "\n\n".join(
            f"USER: {e.get('user', '')}\nBOT: {e.get('reply', '')}"
            for e in recent
        ) or "No recent history."
    
    def add_entry_async(self, user_message, reply_text):
        """Add entry asynchronously - non-blocking"""
        def _add_entry():
            try:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user": str(user_message),
                    "reply": str(reply_text),
                }
                
                with self.lock:
                    self.history.append(entry)
                    
                    # Truncate if needed
                    if len(self.history) > HISTORY_MAX_ITEMS:
                        self.history = self.history[-HISTORY_MAX_ITEMS:]
                    
                    # Update recent cache
                    self.recent_cache = self.history[-self.cache_size:]
                    
                    # Queue for background write
                    history_copy = self.history.copy()
                
                # Non-blocking write queue
                try:
                    self.write_queue.put_nowait(history_copy)
                except queue.Full:
                    print("⚠️ Write queue full, skipping this write")
                
                print(f"✅ History entry added (total: {len(self.history)})")
                
            except Exception as e:
                print(f"❌ Error adding history entry: {e}")
        
        # Execute in background
        self.executor.submit(_add_entry)
    
    def get_history_count(self):
        """Get current history count"""
        with self.lock:
            return len(self.history)
    
    def get_full_history(self):
        """Get full history (for API endpoint)"""
        with self.lock:
            return self.history.copy()
    
    def clear_history(self):
        """Clear all history"""
        with self.lock:
            self.history.clear()
            self.recent_cache.clear()
            # Queue empty list for write
            self.write_queue.put_nowait([])

# Initialize history manager
history_manager = HistoryManager()

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload or "message" not in payload:
            return jsonify({"error": "No message provided"}), 400

        user_message = payload["message"]
        print(f"👤 USER: {user_message}")

        # Get recent context quickly (non-blocking)
        recent_context_str = history_manager.get_recent_context(5)

        # Build prompt
        prompt = f"""
Tum ek AI chatbot ho jo Ajay Kumar Sonkar ka portfolio represent karta hai.
Tum apne bare me sawalon ka jawab Ajay Kumar Sonkar ke taur par doge.
Sirf niche diye gaye JSON data ka use karo.
Agar jawab JSON me na mile to "Mujhe iska jawab data me nahi mila." bolo.

Portfolio Data:
{json.dumps(portfolio, indent=2, ensure_ascii=False)}

Recent History:
{recent_context_str}

User Question: {user_message}

Answer in a friendly, conversational way:
"""

        # Call Gemini model
        try:
            response = model.generate_content(prompt)
            
            reply = "Mujhe iska jawab data me nahi mila."
            
            if hasattr(response, 'text') and response.text:
                reply = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        reply = response.candidates[0].content.parts[0].text.strip()
                except (IndexError, AttributeError) as e:
                    print(f"Error extracting from candidates: {e}")

        except Exception as e:
            print(f"❌ Model generation failed: {e}")
            reply = "Mujhe iska jawab data me nahi mila."

        print(f"🤖 BOT: {reply}")

        # Save in history asynchronously (non-blocking)
        history_manager.add_entry_async(user_message, reply)

        # Return response immediately
        return jsonify({
            "reply": reply, 
            "history_count": history_manager.get_history_count(),
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Chat route error: {e}")
        app.logger.exception("Chat route error")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/history", methods=["GET"])
def get_history():
    """Return full history or last-k via ?k=10"""
    try:
        k = request.args.get("k", type=int)
        full_history = history_manager.get_full_history()
        result = full_history[-k:] if k and k > 0 else full_history
        
        return jsonify({
            "history": result,
            "total_count": len(full_history)
        })
    except Exception as e:
        print(f"❌ History route error: {e}")
        return jsonify({"error": "Failed to get history"}), 500

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear history"""
    try:
        history_manager.clear_history()
        print("🗑️ History cleared successfully")
        return jsonify({"ok": True, "history_count": 0})
    except Exception as e:
        print(f"❌ Failed to clear history: {e}")
        return jsonify({"error": "Failed to clear history"}), 500

@app.route("/test-history", methods=["GET"])
def test_history():
    """Test route to check history functionality"""
    try:
        test_user = "Test message"
        test_reply = "Test reply"
        
        # Add test entry asynchronously
        history_manager.add_entry_async(test_user, test_reply)
        
        return jsonify({
            "status": "success",
            "message": "Test history entry added (async)",
            "history_count": history_manager.get_history_count()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Quick health check without heavy operations"""
    return jsonify({
        "status": "healthy",
        "history_count": history_manager.get_history_count(),
        "portfolio_sections": len(portfolio)
    })

# ---------------------------
# Cleanup on shutdown
# ---------------------------
def cleanup():
    """Cleanup background threads"""
    print("🛑 Shutting down background services...")
    history_manager.write_queue.put(None)  # Shutdown signal
    history_manager.executor.shutdown(wait=True, timeout=5)

import atexit
atexit.register(cleanup)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask app with background history...")
    print(f"📊 Portfolio data loaded with {len(portfolio)} sections")
    print("📚 History loading in background...")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        history_manager.shutdown()
    except Exception as e:
        print(f"❌ Server error: {e}")
        history_manager.shutdown()
    finally:
        cleanup()

