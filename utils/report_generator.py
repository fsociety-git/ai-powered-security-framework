import json

def generate_report(results):
    with open('report.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Report generated.")
