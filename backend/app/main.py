import json

from backend.app.graph import etl_graph


def main():

    user_request = """
    Clean my customer dataset.

    Identify duplicate records, missing values,
    and data quality problems.

    Select appropriate cleaning operations,
    validate the cleaned dataset,
    and load the result into SQLite.
    """

    initial_state = {
        "user_request": user_request,
        "source_path": "customers.csv",
        "target_table": "cleaned_customers",
        "errors": [],
        "execution_log": [],
    }

    result = etl_graph.invoke(
        initial_state,
        config={
            "recursion_limit": 50,
        },
    )

    print("\n========== EXECUTION STATUS ==========")
    print(result.get("status"))

    print("\n========== SUPERVISOR DECISION ==========")
    print(
        json.dumps(
            result.get("decision"),
            indent=2,
        )
    )

    print("\n========== EXECUTION HISTORY ==========")

    for index, event in enumerate(
        result.get("execution_log", []),
        start=1,
    ):
        print(f"\n{index}. {event}")

    print("\n========== FINAL REPORT ==========")
    print(result.get("final_report", "No report generated."))


if __name__ == "__main__":
    main()