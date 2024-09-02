import random

# Example array of names
names = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank"]

# Specify the number of random names you want
number_of_random_names = 3

# Get a random list of names without replacement
random_names_without_replacement = random.sample(names, number_of_random_names)

# Get a random list of names with replacement
random_names_with_replacement = random.choices(names, k=number_of_random_names)

# Print the results
print("Random names without replacement:", random_names_without_replacement)
print("Random names with replacement:", random_names_with_replacement)
