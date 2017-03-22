# Hidden Markov Model for Part of Speech tagging - learning parameters from training file

import sys
import math


def process_args(arglist):
    file_name = arglist[1]
    global num_lines_training

    try:
        with open(file_name, "r") as train_file:
            for line in train_file:
                num_lines_training += 1
                yield line
    except IOError:
        print("Training file not found: {}".format(file_name))


def extract_tag(raw_term):
    return raw_term[-2:]


def extract_token(raw_term):
    return raw_term[:-3]


def init_matrix_key(init_key, target_dict):
    if init_key not in target_dict:
        target_dict[init_key] = {}


# for lists
def update_tag_counts(tag_list, new_tag):
    if new_tag not in tag_list:
        tag_list[new_tag] = 1
    else:
        tag_list[new_tag] += 1


# for dictionaries
def add_to_matrix(key, target_dict):
    if key not in target_dict:
        target_dict[key] = 1
    else:
        target_dict[key] += 1


def process_lines():
    for sentence in process_args(sys.argv):
        sentence_elements = sentence.split()
        for idx, sent_part in enumerate(sentence_elements):  # looping over words in a sentence
            # split the terms into the word and tag
            curr_tag = extract_tag(sent_part)
            curr_word = extract_token(sent_part)

            # modelling priors & transitions by counting
            if idx == 0:
                add_to_matrix(curr_tag, prior_matrix)
            else:
                prev_tag = extract_tag(sentence_elements[idx-1])
                init_matrix_key(prev_tag, transition_matrix)
                add_to_matrix(curr_tag, transition_matrix[prev_tag])
                update_tag_counts(tag_transition_ctr, prev_tag)

            # emission matrix: keys: words; values: (dict-keys: tags; values: occurrence of outer key with given tag
            init_matrix_key(curr_tag, emission_matrix)
            add_to_matrix(curr_word, emission_matrix[curr_tag])
            update_tag_counts(tag_occur_list, curr_tag)


# global variables
emission_matrix = {}
transition_matrix = {}
prior_matrix = {}
tag_occur_list = {}
tag_transition_ctr = {}
num_lines_training = 0

# program start here
process_lines()  # how to start the program? where should it start? how do i make it flow like i want to?

MODEL_FILE_PATH = "hmmmodel.txt"
model_file_object = open(MODEL_FILE_PATH, 'w')

# list of all possible pos tags
all_pos_tags = sorted(list(tag_transition_ctr.keys()))
no_of_pos_tags = len(all_pos_tags)

# writing parameters to model file
for current_tag in all_pos_tags:
    if current_tag not in prior_matrix:
        smoothed_value = math.log(1 / (num_lines_training + no_of_pos_tags))
    else:
        smoothed_value = math.log(
            (prior_matrix[current_tag] + 1) / (num_lines_training + no_of_pos_tags))
    model_file_object.write("{} {} {}\n".format('p', current_tag, str(smoothed_value)))

for pos_tag in tag_transition_ctr.keys():
    for this_tag in all_pos_tags:
        if this_tag not in transition_matrix[pos_tag]:
            smoothed_value = math.log(1/(tag_transition_ctr[pos_tag]+no_of_pos_tags))
        else:
            smoothed_value = math.log((transition_matrix[pos_tag][this_tag] + 1)/(tag_transition_ctr[pos_tag] + no_of_pos_tags))
        model_file_object.write("{} {} {} {}\n".format('t', pos_tag, this_tag, str(smoothed_value)))

for pos_tag in tag_occur_list.keys():
    for term in emission_matrix[pos_tag]:
        ems_prob = math.log(emission_matrix[pos_tag][term] / tag_occur_list[pos_tag])
        model_file_object.write("{} {} {} {}\n".format('e', pos_tag, term, str(ems_prob)))

