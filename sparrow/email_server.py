"""
SMTP server for Reolink camera feed: accepts unauthenticated SMTP on port 25, logs activity,
and saves email attachments to /app/images. Only messages from SENDER_EMAIL to RECIPIENT_EMAIL
are accepted; others are rejected.
"""

import os
import asyncio
import logging
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import AsyncMessage
from aiosmtpd.smtp import SMTP
from email import message_from_bytes, policy
from email.utils import parseaddr

# Configuration Paths
SAVE_DIR = "/app/images"
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "smtp_server.log")

# Setup Logging & Folders
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default Configuration
SENDER_EMAIL = "camera@localnetwork.com"
RECIPIENT_EMAIL = "alerts@localnetwork.com"

# SMTP Server Classes
class CustomSMTP(SMTP):
    async def smtp_AUTH(self, arg):
        logger.info("Authentication skipped, allowing unauthenticated emails.")
        await self.push('235 Authentication successful')
        self.session.authenticated = True

    async def smtp_MAIL(self, arg):
        logger.info(f"MAIL FROM received: {arg}")
        self.session.authenticated = True
        await super().smtp_MAIL(arg)

    async def smtp_EHLO(self, arg):
        logger.info(f"EHLO from camera: {arg}")
        self.session.host_name = arg
        await self.push('250-10.42.0.1')
        await self.push('250-AUTH LOGIN PLAIN')
        await self.push('250 OK')

class EmailHandler(AsyncMessage):
    async def handle_message(self, message):
        logger.info(f"Raw message received")
        email_msg = message_from_bytes(message.as_bytes(), policy=policy.default)
        mail_from = parseaddr(email_msg['From'])[1]
        rcpt_to  = email_msg['To']
        subject  = email_msg['Subject']
        logger.info(f"From: {mail_from}, To: {rcpt_to}, Subject: {subject}")

        if mail_from != SENDER_EMAIL:
            logger.warning("Unauthorized sender: %s", mail_from)
            return '550 Unauthorized sender'
        if RECIPIENT_EMAIL not in rcpt_to:
            logger.warning("Unauthorized recipient: %s", rcpt_to)
            return '550 Unauthorized recipient'

        saved = False
        if email_msg.is_multipart():
            for part in email_msg.iter_parts():
                if part.get_content_disposition() == "attachment":
                    fn = part.get_filename()
                    if fn:
                        path = os.path.join(SAVE_DIR, fn)
                        with open(path, 'wb') as f:
                            f.write(part.get_payload(decode=True))
                        logger.info(f"Saved attachment: {path}")
                        saved = True
        else:
            logger.info("No attachments found.")

        if not saved:
            logger.warning("Email received with no attachments")

        logger.info("Email handled successfully")
        return '250 OK'

class CustomController(Controller):
    def factory(self):
        return CustomSMTP(self.handler, auth_require_tls=False)

# Main
if __name__ == "__main__":
    handler = EmailHandler()
    controller = CustomController(
        handler,
        hostname='0.0.0.0',
        port=25
    )
    controller.start()
    logger.info('SMTP server is running on port 25...')
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
