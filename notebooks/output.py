def format_for_submission(number_array):
    lst = number_array.tolist()
    formatted_numbers = [f"{n:.6f}" for n in lst]
    result = "-".join(formatted_numbers)
    return result
