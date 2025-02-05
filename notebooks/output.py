def format_for_submission(number_array):
    lst = number_array.tolist()
    formatted_numbers = [f"{n:.6f}" for n in lst]
    formatted_numbers = str.replace(formatted_numbers, "1.000000", "0.999999")
    result = "-".join(formatted_numbers)
    return result
