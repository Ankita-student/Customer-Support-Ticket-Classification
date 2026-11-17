"""Create the synthetic support-ticket dataset (data/tickets.csv).

The dataset is created by us (no external dataset is used). Every category has
26 hand-written base descriptions. Each one is used twice with different
openings/closings/noise so the data looks like real, slightly messy tickets.

The file deliberately contains a few duplicates and missing values so that the
data-cleaning step (Task 1) has something real to do.

Run:  python src/generate_dataset.py
"""
import random

import pandas as pd

from config import RAW_CSV, RANDOM_STATE, ensure_dirs

CORES = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset link is not working",
        "My account is locked after too many login attempts",
        "I am not able to sign in with my username and password",
        "The OTP for login is not arriving on my phone",
        "Login page keeps redirecting me back to the sign in screen",
        "I did not receive the password reset email",
        "Unable to log in, it says invalid credentials even though the password is correct",
        "Two factor authentication code is rejected every time I try to sign in",
        "Cannot access my account because I forgot my username",
        "Single sign on is failing and I cannot get into the portal",
        "My login session expires immediately after I sign in",
        "The sign in button does nothing when I click it",
        "I changed my password but still cannot log in with the new one",
        "Account has been temporarily blocked and I cannot sign in",
        "Need help to recover my login credentials",
        "Getting an authentication failed message at the login screen",
        "The captcha on the login page is not accepting my answer",
        "I am locked out of the system since this morning",
        "Security question answer is not accepted when resetting my password",
        "Cannot log in from my new laptop",
        "Login works on mobile but fails on the web portal",
        "Verification code for sign in has expired before I can enter it",
        "Please unlock my user id, I cannot sign in",
        "My password reset keeps saying token expired",
        "Unable to login to the dashboard after the weekend",
    ],
    "Application Error": [
        "The application is showing an error while saving data",
        "I get a 500 internal server error when I submit the form",
        "App crashes every time I open the settings page",
        "Error message appears: something went wrong, please try again",
        "The upload fails with an unexpected exception",
        "Application throws a null pointer error when I click the export button",
        "The page shows a blank screen with an error code after clicking submit",
        "Data is not getting saved and the application shows an error popup",
        "I am seeing a 404 page not found error on the dashboard link",
        "The mobile app closes unexpectedly when I open a ticket",
        "Attachment cannot be uploaded, the system returns an unknown error",
        "Application error occurred while updating the record",
        "The system displays a database connection error",
        "Clicking the save button gives a runtime error",
        "Search feature is broken and returns an error message every time",
        "App froze and then showed a fatal error dialog",
        "I keep getting a bad gateway error when I try to submit the order",
        "The form validation fails with an error even though all the fields are filled",
        "There is a bug in the application, the delete button throws an exception",
        "The software crashed and I lost my unsaved work",
        "Error code 403 appears when I try to open the project page",
        "Application shows an error while importing the CSV file",
        "The notification feature is not working and logs an error",
        "Unexpected error when I try to create a new record",
        "The app stops responding with an error after the latest update",
        "Calendar page gives a server error whenever I open it",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "I need the quarterly revenue report for the finance team",
        "Can you send me last week's attendance report",
        "The export of the customer list to Excel is missing columns",
        "I want a custom report of all closed tickets for this month",
        "Need a summary report of inventory levels",
        "Please schedule the weekly sales summary to be emailed every Monday",
        "The year end report shows wrong totals, please check",
        "How do I download the annual report as PDF",
        "Request for a detailed expense report for the last quarter",
        "Need a report of all users who logged in last month",
        "Can you add a region filter to the sales report",
        "Please generate a profit and loss statement for the last financial year",
        "I would like a dashboard report of customer complaints by month",
        "The scheduled report did not get delivered to my email",
        "Need a report listing all pending orders",
        "Please prepare a monthly summary of employee leave records",
        "Requesting a data extract of all transactions for audit purposes",
        "Report for team productivity is required by end of the week",
        "I need to export the monthly report in CSV format",
        "Please create a report comparing this year sales with last year",
        "Could you share the customer satisfaction report for last quarter",
        "We need a daily collection report for the branch",
        "The report layout is wrong, the charts are missing in the printout",
        "Help me build a report that groups sales by product category",
        "Need the stock movement report for the warehouse",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on the account",
        "I want to change the name on my profile",
        "Need to update my home address in my profile",
        "Please add a new phone number as my alternate contact",
        "How can I change my registered email id",
        "I got married and want to update my surname in my account",
        "Kindly update my date of birth, it was entered incorrectly",
        "I want to close my account permanently",
        "Please change my account type from basic to premium",
        "Need to add a nominee to my account",
        "Update my company name in my profile details",
        "My profile picture cannot be changed, please update it for me",
        "I want to merge my two accounts into one",
        "Please remove my old phone number from the account",
        "Change the language preference on my profile to Hindi",
        "Need to transfer the account ownership to my colleague",
        "Please update the contact person details for our company account",
        "I moved to a new city and need my address updated",
        "I want to deactivate my account for now",
        "Please correct the spelling of my name in my account details",
        "Update my notification preferences to email only",
        "Need to link my new mobile number to the account",
        "Please change the primary email on our organisation account",
        "Add another user to my account with read only access",
        "I want to update the tax id details on my profile",
    ],
    "Performance": [
        "The application is very slow today",
        "Pages take more than a minute to load",
        "The system is lagging badly when I open large files",
        "Dashboard takes forever to load since yesterday",
        "Application response time has become extremely slow",
        "Search results take too long to appear",
        "The software hangs for several seconds every time I click a button",
        "Everything is running slowly during peak hours",
        "Uploading files is taking much longer than usual",
        "The app is sluggish and unresponsive on my laptop",
        "Loading the home screen is very slow on mobile",
        "Report page takes ages to open",
        "The website is timing out and loading extremely slowly",
        "System performance has degraded after the last update",
        "Typing in the editor is delayed by several seconds",
        "It takes a long time to switch between tabs in the application",
        "High latency while saving records, it feels very slow",
        "The application is consuming too much memory and getting slower",
        "Video calls inside the app are lagging and buffering",
        "Our team is facing slow speed whenever we use the portal in the afternoon",
        "Data sync is taking hours instead of minutes",
        "The page keeps buffering and takes too long to respond",
        "Opening the customer list is very slow",
        "Application speed has dropped drastically this week",
        "Slow performance when many users are logged in at the same time",
        "Scrolling through the list is laggy and slow",
    ],
    "Billing Issue": [
        "I was charged twice for my subscription this month",
        "Please send me a copy of my invoice for last month",
        "The amount on my bill is higher than what was agreed",
        "I want a refund for the payment that failed but was deducted",
        "My payment was declined even though my card is valid",
        "Need a GST invoice for the last purchase",
        "I was billed after cancelling my subscription",
        "The discount coupon was not applied to my order total",
        "Please explain the extra charges on my statement",
        "Payment went through but the order still shows as unpaid",
        "How can I change the credit card used for my subscription",
        "I did not receive a refund even after two weeks",
        "The invoice has the wrong company name and tax number",
        "Auto renewal charged me without any notice",
        "Need to update the payment method before the next billing date",
        "There is an unknown transaction on my bill",
        "Please waive the late fee on my latest invoice",
        "My receipt for the annual plan is missing",
        "I upgraded my plan but was still charged the old price",
        "Billing cycle date is wrong on my statement",
        "The credit note for the returned item has not been issued",
        "Money was debited from my bank account but the payment is pending",
        "Why was I charged a renewal fee when I have a lifetime plan",
        "Need a breakdown of the charges on my latest bill",
        "The tax amount on my invoice is calculated incorrectly",
        "Please cancel the pending payment request, I already paid",
    ],
}

OPENERS = ["", "", "", "Hi, ", "Hello team, ", "Dear support, ", "Please help: ",
           "Urgent - ", "Good morning, ", "Hi support, "]
CLOSERS = ["", "", "", " Please help.", " Kindly resolve this soon.", " Thanks.",
           " This is affecting my work.", " Can someone look into it?", " Regards."]

PRIORITY_WEIGHTS = {  # (High, Medium, Low)
    "Login Issue": (0.50, 0.35, 0.15),
    "Application Error": (0.55, 0.35, 0.10),
    "Report": (0.15, 0.45, 0.40),
    "Account Update": (0.10, 0.45, 0.45),
    "Performance": (0.40, 0.45, 0.15),
    "Billing Issue": (0.35, 0.45, 0.20),
}
STATUS_WEIGHTS = {  # (Open, In Progress, Closed)
    "Login Issue": (0.35, 0.20, 0.45),
    "Application Error": (0.40, 0.35, 0.25),
    "Report": (0.20, 0.20, 0.60),
    "Account Update": (0.20, 0.15, 0.65),
    "Performance": (0.40, 0.35, 0.25),
    "Billing Issue": (0.30, 0.25, 0.45),
}

FIRST = ["Aarav", "Priya", "Rahul", "Sneha", "Amit", "Neha", "Vikram", "Anjali", "Rohan",
         "Pooja", "Arjun", "Kavya", "Suresh", "Divya", "Manish", "Ritu", "Karan", "Meera",
         "Sanjay", "Isha", "Deepak", "Nisha", "Rajesh", "Tanvi", "Varun", "Swati"]
LAST = ["Sharma", "Verma", "Das", "Banerjee", "Gupta", "Singh", "Patel", "Roy", "Mehta",
        "Iyer", "Nair", "Ghosh", "Reddy", "Joshi", "Chatterjee"]


def make_text(core, rng):
    opener, closer = rng.choice(OPENERS), rng.choice(CLOSERS)
    if opener.endswith(", ") and not core.startswith("I "):
        core = core[0].lower() + core[1:]
    text = opener + core
    if closer or rng.random() < 0.5:
        text += "."
    text += closer
    # a little real-world noise
    r = rng.random()
    if r < 0.12:
        text = text.lower().replace(",", "").replace(".", "")
    elif r < 0.20:
        text += "!!"
    elif r < 0.28:
        text = text.replace(" ", "  ", 1)
    return text


def build_dataset(seed=RANDOM_STATE):
    rng = random.Random(seed)
    rows = []
    for category, cores in CORES.items():
        seen = set()
        for core in cores:
            for _ in range(2):                 # each base sentence used twice
                text = make_text(core, rng)
                for _try in range(50):
                    if text not in seen:
                        break
                    text = make_text(core, rng)
                seen.add(text)
                rows.append((text, category))
    rng.shuffle(rows)

    records = []
    for i, (text, category) in enumerate(rows, start=1):
        priority = rng.choices(["High", "Medium", "Low"], PRIORITY_WEIGHTS[category])[0]
        status = rng.choices(["Open", "In Progress", "Closed"], STATUS_WEIGHTS[category])[0]
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=rng.randint(0, 300))
        records.append({
            "ticket_id": f"T{i:03d}",
            "customer_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "ticket_description": text,
            "date": date.strftime("%Y-%m-%d"),
            "category": category,
            "priority": priority,
            "status": status,
        })
    df = pd.DataFrame(records)

    # ---- make the data realistically "dirty" -------------------------------
    n = len(df)
    # 1) 5 exact duplicate rows (same ticket_id too)
    exact = df.sample(5, random_state=1)
    # 2) 5 re-submitted tickets: same description, new ticket id
    resub = df.sample(5, random_state=2).copy()
    resub["ticket_id"] = [f"T{n + k + 1:03d}" for k in range(5)]
    df = pd.concat([df, exact, resub], ignore_index=True)

    # 3) missing values (only touch the original rows)
    idx = list(range(n))
    rng.shuffle(idx)
    for i in idx[:4]:
        df.loc[i, "priority"] = None
    for i in idx[4:8]:
        df.loc[i, "status"] = None
    for i in idx[8:10]:
        df.loc[i, "ticket_description"] = None
    df.loc[idx[10], "category"] = None

    return df.sample(frac=1, random_state=3).reset_index(drop=True)


def main():
    ensure_dirs()
    df = build_dataset()
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"Dataset created: {RAW_CSV}")
    print(f"Rows: {len(df)}  |  Categories: {df['category'].nunique()}")


if __name__ == "__main__":
    main()
