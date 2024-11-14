

from collections import defaultdict
import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        basename_emissions = basename + ".emit"
        basename_transition = basename + ".trans"
        with open(basename_emissions, 'r') as f_emit:
            for line in f_emit:
                line = line.strip()
                if not line:
                    continue
                data = line.split(" ")
                state, observation, probability = data[0], data[1], data[2]

                # does key exist in the emissions dictionary
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][observation] = probability

        with open(basename_transition, 'r') as f_transition:
            for line in f_transition:
                line = line.strip()
                if not line:
                    continue
                data = line.split(" ")
                from_state, to_state, probability = data[0], data[1], data[2]
                # does key exist in the transitions dictionary
                if from_state not in self.transitions:
                    self.transitions[from_state] = {}
                self.transitions[from_state][to_state] = probability
        
    def generate(self, n):
        states = list(self.transitions.keys())
        curr_state = random.choice(states)
        while (curr_state == '#'):
             curr_state = random.choice(states) # start in a random initial state and avoid the #
        final_states = []
        observations = []
    
        for _ in range(n):
            # choose the next state based on transition probabilities
            next_states = list(self.transitions[curr_state].keys())
            if "#" in next_states:
                next_states.remove('#')
            weights = list(self.transitions[curr_state].values())

            # turn to floats from strings
            weights = [float(weight) for weight in weights]
            
            curr_state = random.choices(next_states, weights=weights)[0]

            # choose an observation based on emission probabilities and current state
            observations_list = list(self.emissions[curr_state].keys())
            observations_weights = list(self.emissions[curr_state].values())
            observations_weights = [float(weight) for weight in observations_weights]
            curr_observation = random.choices(observations_list, weights=observations_weights)[0]

            final_states.append(curr_state)
            observations.append(curr_observation)

        sequence = Sequence(final_states, observations)
        return sequence, observations
        

    def forward(self, sequence):
        observation_cols = sequence.outputseq  # cols - observations (meow, purr, silent)
        state_rows = list(self.transitions.keys()) # rows - states (happy, grumpy, hungry)
        # print(state_rows)

        matrix = [[0 for _ in range(len(observation_cols) + 1)] for _ in range(len(state_rows))]
        matrix[0][0] = 1 # filling out the # row with 1

        for i in range(1, len(state_rows)): # from 0 to the number of states, fill day 1 (skip the # row)
            matrix[i][1] = float(self.transitions["#"][state_rows[i]]) * float(self.emissions[state_rows[i]][observation_cols[0]])
 
        # Fill in the rest of the matrix (days 2 onward)
        for i in range(2, len(observation_cols) + 1): # starting on day 2
            for j in range(1, len(state_rows)): # for each state
                sum = 0
                for k in range(1, len(state_rows)): # sum over previous states
                    sum += (matrix[k][i - 1] * float(self.transitions[state_rows[k]][state_rows[j]]) * float(self.emissions[state_rows[j]][observation_cols[i - 1]]))
                    print(f"variables in sum: {matrix[k][i - 1]} * {float(self.transitions[state_rows[k]][state_rows[j]])} * {float(self.emissions[state_rows[j]][observation_cols[i - 1]])}")
                    
                
                print(f"sum done, result: {sum} insterted in matrix[{j}][{i}]")
                matrix[j][i] = sum

        max_index = 0
        max_prob = 0       
        for i in range(1, len(state_rows)):
            if matrix[i][len(observation_cols)] > max_prob:
                max_prob = matrix[i][len(observation_cols)]
                max_index = i
    
        return state_rows[max_index]
        






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.



def main():
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("basename", type=str, help="basename for trans & emit")
    parser.add_argument("--generate", type=int, help="generate a sequence of length n", default=None)
    args = parser.parse_args()

    hmm1 = HMM()
    hmm1.load(args.basename)

    if args.generate:
        sequence, observations = hmm1.generate(args.generate)
        # write the observations to a file:
        with open(args.basename + "_sequence.obs", "w") as f:
            for observation in observations:
                f.write(observation + " ")
    
    sequence.outputseq = ["purr", "silent", "silent", "meow", "meow"]
    print(hmm1.forward(sequence))

    

if __name__ == "__main__":
    main()


