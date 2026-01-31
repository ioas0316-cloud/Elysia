def process_data_efficiently(data):
    # This function is more efficient and avoids unnecessary processing.
    result = []
    for item in data:
        if item not in result:
            result.append(item)
    return result

if __name__ == "__main__":
    print("Running sandbox...")
    # placeholder for actual logic
    print(process_data_efficiently([1, 2, 2, 3, 1]))