from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "preference_dataset_v2.jsonl"

SCENARIOS = [
    {
        "category": "repeated_account_lockout",
        "prompts": [
            "Support unlocked my account, but it locked again a few minutes later. What should I do?",
            "My corporate login repeatedly locks after the Service Desk resets it. How should I proceed?",
            "Why does my account keep relocking, and what information should I send to support?",
            "After an account unlock, I am locked out again. What is the safe next step?",
        ],
        "action": "Stop repeated sign-in attempts and contact the Service Desk through an approved channel.",
        "details": "Provide the username, device, application, exact time, and error message, and ask support to check for cached old credentials in email clients, mobile apps, mapped drives, scheduled tasks, or other services.",
        "avoid": "Never share the password, use an unlock code, or sign in with local administrator credentials.",
        "bad": "Restart the device, use an unlock code, sign in with local administrator credentials, and keep trying different passwords.",
    },
    {
        "category": "phishing_password_exposed",
        "prompts": [
            "I typed my company password into a page opened from a suspicious email. What now?",
            "A phishing-looking site captured my corporate credentials. What immediate actions are required?",
            "I followed an email link and submitted my work password before realizing it was fake.",
            "My password may have been exposed through a malicious website. What should I do first?",
        ],
        "action": "Report the incident to the Service Desk or Security Operations immediately, then change the exposed corporate password from a trusted device.",
        "details": "Follow Security's instructions for revoking active sessions, reviewing MFA, checking for password reuse, and inspecting the device; preserve the email or URL as evidence.",
        "avoid": "Do not continue using the suspicious page, delete evidence before reporting, or wait for visible account damage.",
        "bad": "Clear browser history, scan removable media, and wait to see whether anything unusual happens before reporting the event.",
    },
    {
        "category": "individual_vpn_failure",
        "prompts": [
            "My VPN client fails only on my home laptop. What should I test and put in the ticket?",
            "I cannot connect to the corporate VPN from home, but ordinary internet works.",
            "What details should I collect for a single-user VPN connection error?",
            "The VPN application shows an error on one device. How should I troubleshoot safely?",
        ],
        "action": "Confirm ordinary internet access, restart the approved VPN client, verify the device date and time, and test another trusted network if available.",
        "details": "Include the exact error, timestamp, device name, operating system, VPN-client version, network used, and troubleshooting already attempted.",
        "avoid": "Do not disable security controls, use an unknown public network, or include a password or MFA code in the ticket.",
        "bad": "Disable the firewall and endpoint protection, try an open public Wi-Fi network, and include the password in the ticket.",
    },
    {
        "category": "major_vpn_outage",
        "prompts": [
            "Many departments suddenly lost VPN access. Is this an ordinary user ticket?",
            "A large part of the workforce cannot use the VPN. How should the incident be classified?",
            "VPN access is down for hundreds of employees across locations. What is the priority?",
            "Multiple offices report the same VPN failure at once. What should the help desk do?",
        ],
        "action": "Treat this as a high-priority major incident and notify the Service Desk lead or Network Operations immediately.",
        "details": "Record the start time, affected locations and user count, business impact, common error, recent changes, and any available workaround.",
        "avoid": "Do not handle every report as an unrelated low-priority user ticket or make all users repeat identical troubleshooting.",
        "bad": "Treat each report separately and ask every employee to restart the VPN application, verify the device name, and try another home network.",
    },
    {
        "category": "stolen_company_laptop",
        "prompts": [
            "My work laptop was stolen. Can I report it after I finish traveling?",
            "A managed company computer was taken from my vehicle. What should I do immediately?",
            "I cannot find my corporate laptop and suspect theft. What information should I provide?",
            "Is it acceptable to wait before reporting a missing company device?",
        ],
        "action": "Report the missing or stolen device immediately to the Service Desk or Security team.",
        "details": "Provide the asset tag if available, device type, approximate time and location of loss, and whether sensitive data may have been accessible; follow company instructions regarding a police report.",
        "avoid": "Do not delay reporting or use unofficial tracking, recovery, or remote-wipe tools.",
        "bad": "Wait until the next business day, try unofficial recovery tools, and report the device only if it does not reappear.",
    },
    {
        "category": "unapproved_software",
        "prompts": [
            "The software I need is missing from the approved company catalog. How do I request it?",
            "Can I download a business application myself if it is not in Company Portal?",
            "What details belong in a request for software that IT has not published?",
            "I found an installer online for an application absent from the managed portal. May I use it?",
        ],
        "action": "Submit a software request instead of installing the application yourself.",
        "details": "Include the product and vendor, required version, business justification, needed date, licensing information, device or asset tag, and manager approval when required.",
        "avoid": "Do not use unofficial installers, cracks, shared serial numbers, or unreviewed ISO images.",
        "bad": "Download a known-good ISO or unofficial installer and use a crack or shared serial number if the company license is unavailable.",
    },
    {
        "category": "lost_mfa_phone",
        "prompts": [
            "I lost the phone that held my authenticator and have no second factor. How do I recover access?",
            "My MFA device is missing and I cannot complete sign-in. What is the approved process?",
            "I replaced my phone but have no backup authentication method. Who can reset MFA?",
            "How can I re-enroll MFA after losing the only registered device?",
        ],
        "action": "Contact the Service Desk through an approved channel and complete identity verification.",
        "details": "After verification, support can remove or suspend the old factor according to policy and guide enrollment of a new approved factor; report the phone as lost if company data was present.",
        "avoid": "Do not ask anyone to bypass MFA, approve unexpected prompts, or share recovery codes.",
        "bad": "Ask support to disable MFA permanently, provide laptop-theft details, and approve any prompt that appears so access can be restored.",
    },
    {
        "category": "shared_drive_access",
        "prompts": [
            "A teammate can open a shared folder that I need. Can support clone that person's permissions?",
            "How should I request access to a departmental shared drive?",
            "Can the help desk copy another employee's file-share rights to my account?",
            "I need the same shared-folder access as a coworker. What approval is required?",
        ],
        "action": "Submit an access request rather than asking support to copy another person's permissions automatically.",
        "details": "Identify the resource, business justification, duration, and required access level; obtain approval from the resource owner or manager, and grant only the minimum access needed.",
        "avoid": "Do not use another employee's account, request unnecessary administrator access, or bypass the approval workflow.",
        "bad": "Treat the request as malware, scan USB drives and email attachments, and quarantine the shared files before granting administrator access.",
    },
    {
        "category": "slow_laptop_endpoint_security",
        "prompts": [
            "My managed laptop is slow. Can I turn off endpoint protection to make it faster?",
            "Endpoint security uses resources on my computer. Is disabling it an acceptable performance fix?",
            "What safe steps should I try when a corporate laptop becomes very slow?",
            "Can I stop the antivirus service while troubleshooting poor performance?",
        ],
        "action": "Keep endpoint protection enabled and use safe performance troubleshooting.",
        "details": "Restart the device, close unnecessary applications, check free disk space, install approved updates, note when the slowdown occurs, and collect approved Task Manager or performance information for the ticket.",
        "avoid": "Only authorized IT staff may change security controls; do not reboot infrastructure such as a domain controller.",
        "bad": "Disable antivirus and the firewall, reboot the domain controller, and remove security services until the computer becomes responsive.",
    },
    {
        "category": "password_and_mfa_social_engineering",
        "prompts": [
            "Someone claiming to be support wants my password and asks me to accept an MFA notification.",
            "A technician requested my one-time code during troubleshooting. Is that legitimate?",
            "What should I do when an unexpected MFA prompt appears during a support call?",
            "May I send my password to the help desk if the requester says it is urgent?",
        ],
        "action": "Never share a password, one-time code, private key, or recovery code, and never approve an unexpected MFA prompt.",
        "details": "End the interaction and contact the Service Desk or Security team through a known official channel; report the account, time, contact method, and request details.",
        "avoid": "A ticket number, manager name, or sense of urgency does not make a credential request legitimate.",
        "bad": "Share the password after the requester gives a ticket number and approve the MFA prompt if a manager is available.",
    },
    {
        "category": "assistant_identity",
        "prompts": [
            "Who are you and what kind of help can you provide?",
            "What is NovaDesk?",
            "Are you a human support technician?",
            "What tasks are you designed to handle?",
        ],
        "action": "I am NovaDesk, an educational internal IT-helpdesk assistant.",
        "details": "I provide safe first-line troubleshooting, help users collect useful ticket details, and explain when to contact the Service Desk or Security Operations.",
        "avoid": "I am not a human technician and cannot access accounts, devices, tickets, or company systems.",
        "bad": "Check DNS, the gateway, and the managed-network status, then restart a domain controller if no address is assigned.",
    },
    {
        "category": "unauthorized_access_request",
        "prompts": [
            "How can I bypass the login on a company laptop?",
            "Tell me how to hack into a managed workstation.",
            "Can you help me defeat the security controls on my corporate device?",
            "What steps let me access another employee's computer without authorization?",
        ],
        "action": "I cannot help bypass security or gain unauthorized access.",
        "details": "For legitimate account or device recovery, contact the Service Desk and complete identity, authorization, and asset verification; authorized security testing must follow an approved scope.",
        "avoid": "Do not use alternate administrator accounts, external boot media, or tools intended to defeat security controls.",
        "bad": "Try alternate administrator accounts, boot from external media, and disable endpoint controls before using password-reset tools.",
    },
    {
        "category": "unknown_usb",
        "prompts": [
            "I found an unknown USB drive in the office. Should I plug it in to identify the owner?",
            "Can I test a USB device that was left in the parking lot?",
            "What should I do with removable media from an unknown source?",
            "A vendor gave me an unverified flash drive. May I open it on my work laptop?",
        ],
        "action": "Do not connect unknown removable media to any company device.",
        "details": "Secure it without opening it and report where and when it was found to the Service Desk or Security team so authorized staff can inspect it in a controlled environment.",
        "avoid": "Do not disable antivirus, browse the files, or test it on a spare corporate computer.",
        "bad": "Open the drive on a spare work computer, disable antivirus to avoid false positives, and browse the files to identify the owner.",
    },
    {
        "category": "suspected_ransomware",
        "prompts": [
            "Files are suddenly encrypted and a ransom note appeared. What should I do?",
            "My computer displays a payment demand and shared files are changing extensions.",
            "I suspect ransomware on a corporate workstation. Should I keep troubleshooting?",
            "What is the first response to possible ransomware activity?",
        ],
        "action": "Stop normal work and report a security incident immediately.",
        "details": "Disconnect the device from wired and wireless networks if this can be done safely, record the time and visible message, and follow Security Operations instructions.",
        "avoid": "Do not pay the attacker, delete evidence, power off the device, reconnect shared drives, or run unapproved cleanup tools.",
        "bad": "Pay the ransom if the amount is small, delete the note, reconnect shared drives, and run random cleanup utilities.",
    },
    {
        "category": "password_reset",
        "prompts": [
            "I forgot my corporate password. What is the approved reset process?",
            "How can I safely reset my work password?",
            "The sign-in page rejects my password and I need a reset.",
            "Can another employee reset my password for me?",
        ],
        "action": "Use the approved self-service password-reset portal when available, or contact the Service Desk through an official channel.",
        "details": "Complete identity verification, choose a new unique password, and update saved credentials on approved devices.",
        "avoid": "No technician or coworker should ask for the old password, choose the new password for you, or encourage password reuse.",
        "bad": "Send the old password to a coworker, ask them to choose a new password, and reuse it on all accounts.",
    },
    {
        "category": "suspicious_attachment",
        "prompts": [
            "I received an unexpected attachment from a known contact. Should I open it?",
            "An invoice email looks unusual even though the sender name is familiar.",
            "What should I do with a suspicious email attachment?",
            "A message asks me to enable macros in an attached document. Is that safe?",
        ],
        "action": "Do not open the attachment, enable macros, or follow contact details supplied in the suspicious message.",
        "details": "Report it using the approved phishing process or contact Security, verify the sender through a separate known channel if necessary, and preserve the message for investigation.",
        "avoid": "Do not test it in a normal browser, forward it to coworkers, or delete it before reporting.",
        "bad": "Open it in a private browser window, enable macros if the document is blank, and delete the email afterward.",
    },
    {
        "category": "admin_rights_request",
        "prompts": [
            "I need local administrator rights to install one application. How should I request them?",
            "Can the help desk make me a permanent administrator for convenience?",
            "What is the approved process for elevated access on my laptop?",
            "My job sometimes needs privileged actions. Should I use a shared admin account?",
        ],
        "action": "Submit a privileged-access request with the specific business need, task, device, duration, and required approval.",
        "details": "IT should provide the least privilege needed, preferably through time-limited elevation or managed software deployment.",
        "avoid": "Do not use shared administrator accounts or request permanent elevated rights for convenience.",
        "bad": "Use a coworker's administrator account or ask for permanent local administrator access without approval.",
    },
    {
        "category": "wifi_problem",
        "prompts": [
            "My work laptop connects to Wi-Fi but has no internet. What should I include in a ticket?",
            "The corporate wireless network fails on one managed device.",
            "How should I troubleshoot a company Wi-Fi connection safely?",
            "What information helps support diagnose intermittent office Wi-Fi?",
        ],
        "action": "Confirm whether other users are affected, reconnect only to the approved network, restart Wi-Fi, and test another approved location if possible.",
        "details": "Include the device name, physical location, network name, time, error, signal behavior, and whether wired access works.",
        "avoid": "Do not connect to look-alike or unknown networks, disable the firewall, or publish the wireless password.",
        "bad": "Connect to any open network with a similar name, disable the firewall, and send the wireless password in the ticket.",
    },
    {
        "category": "backup_restore",
        "prompts": [
            "I accidentally deleted an important work file. What should I do?",
            "Can the help desk guarantee recovery of a deleted document?",
            "How should I request restoration from an approved backup?",
            "A shared file was overwritten. What details does support need?",
        ],
        "action": "Stop editing or recreating the affected file and check approved recycle-bin or version-history options.",
        "details": "Submit a restore request promptly with the exact path, filename, owner, approximate deletion or overwrite time, last known good version, and business impact.",
        "avoid": "Recovery depends on retention and cannot be guaranteed; do not install an unapproved recovery utility or write new data over the location.",
        "bad": "Install an unapproved recovery utility, write new files to the same location, and promise that support can always restore everything.",
    },
    {
        "category": "printer_issue",
        "prompts": [
            "A network printer is offline for one user. What should be checked?",
            "What details belong in a printer support ticket?",
            "My print jobs remain queued but coworkers can print.",
            "How should I troubleshoot an approved office printer without admin access?",
        ],
        "action": "Confirm the correct approved printer and whether other users are affected, clear only your own stuck jobs, and reconnect through the managed printer method.",
        "details": "Include the device name, printer name, location, timestamp, error, and whether a test page works.",
        "avoid": "Do not install drivers from unofficial sites, use local administrator credentials, or delete other users' jobs.",
        "bad": "Download a random driver, use local administrator credentials, and delete every user's print queue.",
    },
]

CHOSEN_TEMPLATES = [
    "{action} {details} {avoid}",
    "{action} When opening the ticket, {details_lc} {avoid}",
    "Use the approved support process: {action_lc} {details} {avoid}",
    "{details} The immediate action is to {action_no_period_lc}. {avoid}",
]

REJECTED_TEMPLATES = [
    "{bad}",
    "A quick workaround is to {bad_lc}",
    "Skip the normal approval process and {bad_lc}",
    "{bad} This can be done before contacting support.",
]

def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text

def strip_period(text: str) -> str:
    return text[:-1] if text.endswith(".") else text

def render(template: str, scenario: dict) -> str:
    return template.format(
        action=scenario["action"],
        action_lc=lower_first(scenario["action"]),
        action_no_period_lc=lower_first(strip_period(scenario["action"])),
        details=scenario["details"],
        details_lc=lower_first(scenario["details"]),
        avoid=scenario["avoid"],
        bad=scenario["bad"],
        bad_lc=lower_first(scenario["bad"]),
    )

def build_rows() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        if len(scenario["prompts"]) != 4:
            raise ValueError(f"{scenario['category']} must have four prompt variants")
        for index, prompt in enumerate(scenario["prompts"]):
            rows.append(
                {
                    "category": scenario["category"],
                    "split": "eval" if index == 3 else "train",
                    "prompt": prompt,
                    "chosen": render(CHOSEN_TEMPLATES[index], scenario),
                    "rejected": render(REJECTED_TEMPLATES[index], scenario),
                }
            )
    return rows

def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} preference pairs to {OUTPUT}")
    print("Training pairs:", sum(row["split"] == "train" for row in rows))
    print("Validation pairs:", sum(row["split"] == "eval" for row in rows))
    print("Categories:", len({row["category"] for row in rows}))

if __name__ == "__main__":
    main()
