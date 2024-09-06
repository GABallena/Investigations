import random
import math

# Define your trimming tools and parameters
trimming_tools = ['fastp', 'trimmomatic', 'cutadapt', 'bbduk', 'sickle']
parameters = {'quality_threshold': [20, 25, 30], 'min_length': [50, 75, 100]}

def calculate_likelihood(tool_combination):
    # Dummy function to calculate likelihood based on info density
    # You'd replace this with real calculations using your data
    info_density = random.random()  # placeholder for actual info density calculation
    return info_density

def propose_new_state(current_state):
    # Randomly change one of the tools or parameters
    new_state = current_state.copy()
    new_state['tool'] = random.choice(trimming_tools)
    new_state['quality_threshold'] = random.choice(parameters['quality_threshold'])
    new_state['min_length'] = random.choice(parameters['min_length'])
    return new_state

def metropolis_hastings_chain(start_state, steps):
    current_state = start_state
    current_likelihood = calculate_likelihood(current_state)
    
    for _ in range(steps):
        proposed_state = propose_new_state(current_state)
        proposed_likelihood = calculate_likelihood(proposed_state)
        
        # Accept the new state if the likelihood is better
        if proposed_likelihood > current_likelihood:
            current_state = proposed_state
            current_likelihood = proposed_likelihood
        else:
            # Otherwise, accept with a certain probability
            acceptance_prob = math.exp(proposed_likelihood - current_likelihood)
            if random.random() < acceptance_prob:
                current_state = proposed_state
                current_likelihood = proposed_likelihood

        print(f"Step: {_:3}, State: {current_state}, Likelihood: {current_likelihood:.3f}")

# Initialize starting states for 4 chains
start_state = {'tool': random.choice(trimming_tools),
               'quality_threshold': random.choice(parameters['quality_threshold']),
               'min_length': random.choice(parameters['min_length'])}

# Run chains in parallel
for i in range(4):
    print(f"Running Chain {i + 1}")
    metropolis_hastings_chain(start_state, 100)  # 100 steps for simplicity
    print("\n")
