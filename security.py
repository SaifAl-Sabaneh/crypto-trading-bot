import os
import sys
import time
import logging
import tempfile
import urllib.request
import config

# =====================================================================
# SYSTEM LOGGER CONFIGURATION WITH CREDENTIAL MASKING
# =====================================================================
class CredentialMaskingFormatter(logging.Formatter):
    """
    Formatter that ensures secrets, API keys, and sensitive tokens 
    are masked (hidden) in both console and file log outputs.
    """
    def __init__(self, fmt=None, datefmt=None, secrets=None):
        super().__init__(fmt, datefmt)
        self.secrets = secrets or []

    def format(self, record):
        message = super().format(record)
        # Mask secrets if any are loaded in config
        for secret in self.secrets:
            if secret and len(secret) > 4:
                message = message.replace(secret, f"***MASKED_SECRET_{secret[-4:]}***")
        return message

def setup_logger():
    """Sets up a robust dual-output logging pipeline."""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Secrets to mask
    secrets = [config.API_KEY, config.SECRET_KEY]
    
    formatter = CredentialMaskingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        secrets=secrets
    )
    
    logger = logging.getLogger("TradingBot")
    logger.setLevel(log_level)
    logger.handlers.clear()  # Clear existing handlers
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (Atomic Append)
    try:
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize log file handler: {e}. Logging to console only.")
        
    return logger

# Initialize system logger
logger = setup_logger()

# =====================================================================
# ATOMIC I/O OPERATIONS (FAIL-PROOF STORAGE)
# =====================================================================
def safe_atomic_write(filepath, content):
    """
    Writes content to a file atomically.
    Ensures that if the bot crashes or loses power during writing,
    the existing file is not corrupted.
    """
    dir_name = os.path.dirname(filepath) or "."
    base_name = os.path.basename(filepath)
    
    # Create temp file in same directory to ensure atomic rename works on Windows/Unix
    try:
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp', encoding='utf-8') as tf:
            tf.write(content)
            temp_path = tf.name
            
        # Atomic replacement: rename temp file to target path
        if os.path.exists(filepath):
            os.replace(temp_path, filepath)
        else:
            os.rename(temp_path, filepath)
        return True
    except Exception as e:
        logger.error(f"Atomic file write failed for {filepath}: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

# =====================================================================
# RESILIENT NETWORK RETRY DECORATOR (FAIL-PROOF API INTERACTION)
# =====================================================================
def network_retry(retries=3, backoff_factor=2.0):
    """
    Decorator that retries network operations with exponential backoff.
    Catches common connection/network exceptions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        logger.error(f"Network operation '{func.__name__}' failed after {retries} attempts: {e}")
                        raise e
                    logger.warning(f"Network error in '{func.__name__}' (attempt {attempt}/{retries}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            return None
        return wrapper
    return decorator

# =====================================================================
# HEALTH MONITOR (PRE-FLIGHT SECURITY CHECKS)
# =====================================================================
class HealthMonitor:
    """
    Checks connection, credential storage, and disk space
    before allowing bot execution.
    """
    @staticmethod
    def check_network():
        """Verifies internet connection by trying to reach standard domains."""
        urls = ["https://query1.finance.yahoo.com", "https://www.google.com"]
        for url in urls:
            try:
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                urllib.request.urlopen(req, timeout=3.0)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def check_disk_space(min_mb=50):
        """Checks if there is enough space on local disk for database/logs."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_mb = free / (1024 * 1024)
            return free_mb > min_mb
        except Exception:
            return True # Default to true if os command fails

    @classmethod
    def run_health_checks(cls):
        """Runs pre-flight checks and returns system readiness status."""
        logger.info("Initializing pre-flight security checks...")
        
        # 1. Check Network
        if not cls.check_network():
            logger.critical("No internet connection detected. Network checks failed.")
            return False
        logger.info("Health Check: Internet Connection [OK]")
        
        # 2. Check Disk Space
        if not cls.check_disk_space():
            logger.error("Disk space critical (less than 50MB remaining). Logs may fail.")
            return False
        logger.info("Health Check: System Disk Space [OK]")
        
        # 3. Check Credentials
        if not config.API_KEY or not config.SECRET_KEY:
            if config.IS_SANDBOX:
                logger.warning("No API credentials found. Sandbox paper-trading active [SAFE].")
            else:
                logger.error("CRITICAL: IS_SANDBOX=False but no API credentials found. Execution halted for security.")
                return False
        else:
            logger.info("Health Check: API Credentials Verified [OK]")
            
        logger.info("All pre-flight checks passed successfully. System Ready.")
        return True

def send_push_notification(message):
    """
    Sends a push notification to Discord and/or Telegram if configured.
    Runs silently in a try-except block so network failures on webhook alerts never crash the bot.
    """
    # 1. Discord Webhook Integration
    if config.DISCORD_WEBHOOK_URL:
        try:
            import json
            data = {"content": f"🤖 **Trading Bot Alert**:\n{message}"}
            req = urllib.request.Request(
                config.DISCORD_WEBHOOK_URL,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
            )
            urllib.request.urlopen(req, timeout=5.0)
            logger.info("Discord notification sent successfully.")
        except Exception as e:
            logger.warning(f"Failed to send Discord notification: {e}")

    # 2. Telegram Bot Integration
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        try:
            import json
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            # Format markdown carefully to avoid parsing errors
            data = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": f"🤖 *Trading Bot Alert*:\n{message}",
                "parse_mode": "Markdown"
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
            )
            urllib.request.urlopen(req, timeout=5.0)
            logger.info("Telegram notification sent successfully.")
        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")

