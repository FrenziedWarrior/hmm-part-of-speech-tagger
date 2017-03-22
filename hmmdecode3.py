# Hidden Markov Model decoder
# Extract transition and emission probabilities from hmmmodel.txt
# Writes out tagged terms to hmmoutput.txt
# Set up the dynamic programming Viterbi algorithm
# Maintain backpointers

import sys

prior_params = {}
trans_params = {}
emiss_params = {}
prob_matrix = {}
backptr_matrix = {}

MODEL_FILE = "hmmmodel.txt"
OUTPUT_FILE = "hmmoutput.txt"


def get_emission_value(target_state, target_token):
    # if obs_start exists in emissionMatrix(currentState)
    if target_token not in emiss_params[target_state]:
        return 0.0
    else:
        return emiss_params[target_state][target_token]


# program starts here
with open(MODEL_FILE, 'r') as hmm_model:
    for line in hmm_model:  # for each line in the model file
        params = line.split()   # split the parameters
        matrix_type = params[0]  # type of matrix to fill
        # decide how to store the parameters in the matrices depending on type
        if matrix_type == 'p':
            prior_params[params[1]] = float(params[2])
        elif matrix_type == 't':
            if params[1] not in trans_params:
                trans_params[params[1]] = {}
            trans_params[params[1]][params[2]] = float(params[3])
        elif matrix_type == 'e':
            if params[1] not in emiss_params:
                emiss_params[params[1]] = {}
            emiss_params[params[1]][params[2]] = float(params[3])

# alphabetically sorted list of all POS tags
pos_tags = sorted(list(prior_params.keys()))
out_file = open(OUTPUT_FILE, 'w')

with open(sys.argv[1], 'r') as dev_set:
    for line in dev_set:
        sentence = line.split()
        length_seq = len(sentence)

        # initial state transitions
        unknown_first_token_flag = True
        valid_initial_states = []
        for p in pos_tags:
            if get_emission_value(p, sentence[0]) != 0.0:
                valid_initial_states.append(p)
                unknown_first_token_flag = False

        for state in pos_tags:
            prob_matrix[state] = []
            backptr_matrix[state] = []
            if (not unknown_first_token_flag) and (state not in valid_initial_states):
                prob_matrix[state].append(0)
                backptr_matrix[state].append("--")
            else:
                emission_start = get_emission_value(state, sentence[0])
                prob_matrix[state].append(prior_params[state] + emission_start)
                backptr_matrix[state].append("Q0")

        # recursion for remaining observations in sentence
        for idx, token in enumerate(sentence[1:], start=1):  # for each token in the sequence
            # check if the token appears in the training set at all
            unknown_token_flag = True
            valid_states = []
            for p in pos_tags:
                if get_emission_value(p, token) != 0.0:
                    valid_states.append(p)
                    unknown_token_flag = False

            # if token is unknown, proceed
            # if token is known, but curr_state is not in valid_states, append 0 probability, append -- backptr
            # if token is known, but curr_state is in valid_states, proceed

            incoming_states = []
            for prev_state in pos_tags:
                if prob_matrix[prev_state][idx-1] != 0:
                    incoming_states.append(prev_state)

            for curr_state in pos_tags:
                if (not unknown_token_flag) and (curr_state not in valid_states):
                    prob_matrix[curr_state].append(0)
                    backptr_matrix[curr_state].append('--')
                else:  # if ((not unknown_token_flag) and (curr_state in valid_states)) or (unknown_token_flag):
                    max_prob_value = -float("inf")
                    max_backptr_value = -float("inf")
                    emission_value = get_emission_value(curr_state, token)

                    for prev_state in incoming_states:
                        transition_value = trans_params[prev_state][curr_state]
                        prev_state_probability = prob_matrix[prev_state][idx-1]
                        candidate_prob = prev_state_probability + transition_value + emission_value
                        candidate_bptr = prev_state_probability + transition_value
                        # best probability value
                        if max_prob_value < candidate_prob:
                            max_prob_value = candidate_prob
                        # best backpointer value
                        if max_backptr_value < candidate_bptr:
                            max_backptr_value = candidate_bptr
                            back_pointer_state = prev_state
                    prob_matrix[curr_state].append(max_prob_value)
                    backptr_matrix[curr_state].append(back_pointer_state)

        # ref_line_1 = [term[-2:] for term in ref_set.readline().split()]

        temp_max = -float("inf")
        for state in pos_tags:
            end_prob = prob_matrix[state][-1]
            if (end_prob != 0) and (temp_max<end_prob):
                temp_max = end_prob
                most_probable_state_end = state

        final_state_seq = [most_probable_state_end]

        current_state = backptr_matrix[most_probable_state_end][-1]
        for idx in range(length_seq-2, -1, -1):
            final_state_seq.append(current_state)
            current_state = backptr_matrix[current_state][idx]

        for idx, pos_tag in enumerate(final_state_seq[:0:-1]):
            out_file.write("{}/{} ".format(sentence[idx], pos_tag))
        out_file.write("{}/{}\n".format(sentence[-1], final_state_seq[0]))

