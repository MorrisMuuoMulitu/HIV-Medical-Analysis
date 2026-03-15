"""
Email Report Scheduler
======================

Schedule and send automated PDF reports via email.
"""

import smtplib
import os
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmailReportScheduler:
    """Schedule and send PDF reports via email."""
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None
    ):
        """
        Initialize email scheduler.
        
        Args:
            smtp_server: SMTP server address (e.g., smtp.gmail.com)
            smtp_port: SMTP port (default: 587 for TLS)
            sender_email: Sender email address
            sender_password: Sender email password or app password
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', 587))
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        
        self.recipients: List[str] = []
        self.scheduled_reports: List[Dict[str, Any]] = []
    
    def add_recipient(self, email: str) -> 'EmailReportScheduler':
        """Add a recipient to the email list."""
        self.recipients.append(email)
        return self
    
    def add_recipients(self, emails: List[str]) -> 'EmailReportScheduler':
        """Add multiple recipients."""
        self.recipients.extend(emails)
        return self
    
    def schedule_report(
        self,
        report_path: str,
        subject: str,
        body: str,
        send_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Schedule a report for sending.
        
        Args:
            report_path: Path to the PDF report
            subject: Email subject
            body: Email body text
            send_date: When to send (None = send immediately)
            
        Returns:
            Scheduled report metadata
        """
        report_info = {
            'report_path': report_path,
            'subject': subject,
            'body': body,
            'send_date': send_date or datetime.now(),
            'recipients': self.recipients.copy(),
            'status': 'pending'
        }
        
        self.scheduled_reports.append(report_info)
        logger.info(f"Scheduled report: {report_path} for {send_date or 'immediate'}")
        
        return report_info
    
    def send_report(
        self,
        report_path: str,
        subject: str,
        body: str,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send a report immediately.
        
        Args:
            report_path: Path to the PDF report
            subject: Email subject
            body: Email body text
            recipients: List of recipient emails (uses default if None)
            
        Returns:
            True if sent successfully
        """
        if not self.sender_email or not self.sender_password:
            logger.error("Email credentials not configured")
            return False
        
        recipients = recipients or self.recipients
        if not recipients:
            logger.error("No recipients specified")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Attach PDF
            if Path(report_path).exists():
                with open(report_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{Path(report_path).name}"'
                    )
                    msg.attach(part)
            else:
                logger.warning(f"Report file not found: {report_path}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Report sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send report: {e}")
            return False
    
    def send_scheduled_reports(self) -> Dict[str, int]:
        """
        Send all scheduled reports that are due.
        
        Returns:
            Dictionary with counts of sent/failed reports
        """
        now = datetime.now()
        results = {'sent': 0, 'failed': 0, 'pending': 0}
        
        for report in self.scheduled_reports:
            if report['status'] == 'pending' and report['send_date'] <= now:
                success = self.send_report(
                    report['report_path'],
                    report['subject'],
                    report['body'],
                    report['recipients']
                )
                
                if success:
                    report['status'] = 'sent'
                    report['sent_at'] = now
                    results['sent'] += 1
                else:
                    report['status'] = 'failed'
                    results['failed'] += 1
            elif report['status'] == 'pending':
                results['pending'] += 1
        
        return results
    
    def create_email_template(
        self,
        report_type: str,
        report_date: str,
        key_metrics: Dict[str, str]
    ) -> tuple[str, str]:
        """
        Create an HTML email template.
        
        Args:
            report_type: Type of report (summary, detailed, executive)
            report_date: Report generation date
            key_metrics: Dictionary of key metrics to display
            
        Returns:
            Tuple of (subject, html_body)
        """
        subject = f"HIV/AIDS Analytics Report - {report_date}"
        
        # Build metrics table
        metrics_rows = ""
        for metric, value in key_metrics.items():
            metrics_rows += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{metric}</td>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{value}</td>
            </tr>
            """
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .header {{ 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 20px; 
                    text-align: center;
                }}
                .content {{ padding: 20px; }}
                .metrics {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                }}
                .metrics th {{ 
                    background-color: #667eea; 
                    color: white; 
                    padding: 10px;
                }}
                .footer {{ 
                    background-color: #f4f4f4; 
                    padding: 15px; 
                    text-align: center; 
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HIV/AIDS Analytics Report</h1>
                <p>{report_type.title()} Report | {report_date}</p>
            </div>
            
            <div class="content">
                <p>Dear Stakeholder,</p>
                
                <p>Please find attached the {report_type} HIV/AIDS analytics report generated on {report_date}.</p>
                
                <h3>Key Metrics Summary</h3>
                <table class="metrics">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {metrics_rows}
                </table>
                
                <p>This report provides comprehensive insights into:</p>
                <ul>
                    <li>Patient demographics and trends</li>
                    <li>Treatment outcomes and efficacy</li>
                    <li>Viral suppression rates</li>
                    <li>Key recommendations for improvement</li>
                </ul>
                
                <p>Please review the attached PDF for detailed analysis and visualizations.</p>
                
                <p>Best regards,<br>
                <strong>HIV/AIDS Analytics Platform</strong></p>
            </div>
            
            <div class="footer">
                <p>This is an automated message from the HIV/AIDS Analytics Platform.</p>
                <p>For questions or support, please contact your healthcare data team.</p>
                <p>&copy; {datetime.now().year} HIV/AIDS Analytics Platform. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        return subject, html_body


# Convenience function for quick email sending
def send_report_email(
    report_path: str,
    recipients: List[str],
    report_type: str = "summary",
    key_metrics: Optional[Dict[str, str]] = None
) -> bool:
    """
    Send a report via email.
    
    Args:
        report_path: Path to the PDF report
        recipients: List of recipient emails
        report_type: Type of report
        key_metrics: Key metrics to include in email
        
    Returns:
        True if sent successfully
    """
    scheduler = EmailReportScheduler()
    scheduler.add_recipients(recipients)
    
    report_date = datetime.now().strftime("%B %d, %Y")
    key_metrics = key_metrics or {}
    
    subject, html_body = scheduler.create_email_template(
        report_type, report_date, key_metrics
    )
    
    return scheduler.send_report(report_path, subject, html_body, recipients)
