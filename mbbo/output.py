import csv
import numpy as np

class ScenarioResult():
    def __init__(self, description, result_list):
        self.description = description
        self.result_list = result_list

class TestResults():

    def __init__(self, scenario_results):
        self.scenario_results = scenario_results

    def describe_best_case(self):
        if not self.scenario_results:
            return "No scenarios available."

        best_scenario = None
        best_avg = float('inf')

        for scenario in self.scenario_results:
            result_array = np.array(scenario.result_list, dtype=np.float64)

            # Disqualify if any value is NaN
            if np.isnan(result_array).any():
                continue

            # Filter nonzero elements
            nonzero_elements = result_array[result_array != 0]

            # If no nonzero elements remain, skip this scenario
            if len(nonzero_elements) == 0:
                continue

            # Compute average of nonzero elements
            avg_value = np.mean(nonzero_elements)

            # Update best scenario if this one has a lower average
            if avg_value < best_avg:
                best_avg = avg_value
                best_scenario = scenario

        # Handle case where no valid scenarios exist
        if best_scenario is None:
            return "Best: no valid scenarios found (all disqualified due to NaN or zero-only results)."

        return f"Best: {best_scenario.description} with avg iterations: {best_avg:.2f}"


def format_for_submission(number_array):
    lst = number_array.tolist()
    formatted_numbers = [f"{n:.6f}" for n in lst]
    result = "-".join(formatted_numbers)
    result = str.replace(result, "1.000000", "0.999999")
    return result

def output_results_csv(test_results, filename="out"):
    headers = ["Description"]
    for i in range(len(test_results.scenario_results[0].result_list)):
        headers.append("Function " + str(i))
    with open(f'{filename}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for sr in test_results.scenario_results:
            row = []
            row.append(sr.description)
            row.extend(sr.result_list)
            writer.writerow(row)
        writer.writerow([test_results.describe_best_case()])

# Todo: wrap these 3 in class
def start_results_csv(number_of_test_functions, filename="out"):
    headers = ["Description"]
    for i in range(number_of_test_functions):
        headers.append("Function " + str(i))

    f_name = f'{filename}.csv'
    with open(f_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return f_name

def write_test_result(filename, scenario_result:ScenarioResult):
    """Writes a single row of test results to the CSV file."""
    row = [scenario_result.description] + scenario_result.result_list

    with open(filename, 'a', newline='') as f:  # Append mode

        writer = csv.writer(f)
        writer.writerow(row)

def write_best_case_description(filename, test_results:TestResults):
    """Writes the best case description at the end of the CSV file."""
    with open(filename, 'a', newline='') as f:  # Append mode
        writer = csv.writer(f)
        writer.writerow([test_results.describe_best_case()])