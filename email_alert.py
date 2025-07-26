import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(to_email, stock, signal, ltp, cluster_high, cluster_low):
    # Gmail SMTP setup
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "sampathskh@gmail.com"  # Use your Gmail address
    sender_password = "zeda mdbq dwag xeyx"  # Use an app password, not your Gmail password

    subject = f"[Friday Cluster Alert] {stock} - {signal}"
    body = f"""
Stock: {stock}\n
Signal: {signal}\n
Last Traded Price: {ltp}\n
Friday Cluster High: {cluster_high}\n
Friday Cluster Low: {cluster_low}\n
"""

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")
