def format_for_submission(number_array):
    lst = number_array.tolist()
    formatted_numbers = [f"{n:.6f}" for n in lst]
    result = "-".join(formatted_numbers)
    result = str.replace(result, "1.000000", "0.999999")
    return result

def output_results_csv(number_of_functions, results, filename="out"):
    headers = ["Lengthscale", "Acq kappa"]
    for i in range(number_of_functions):
        headers.append("Function " + str(i))
    with open(f'{filename}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
