import numpy as np

class Score:
    def __init__(self):
        self.prob_blank = 0
        self.prob_non_blank = 0
        self.prob_total = 0

class Beams:
    """
    A Class for a set of beams (only to be used for "beam_search_decoder"). 
    Beams are stored as dictionaries where the key is a label sequence and the 
    value is probability of the seq.
    """
    def __init__(self):
        self.beams = {}

    def get_top_beams(self):
        '''
        Returns the beams sorted by scored in reverse order.

        '''
        sorted_tuple = sorted(self.beams.items(), key=lambda kv:kv[1].prob_total, reverse=True)
        return [s_tuple[0] for s_tuple in sorted_tuple]

    def new_beam(self,labeling):
        '''

        Add a new beam if it's doesn't exist.

        '''
        if labeling not in self.beams:
            self.beams[labeling] = Score()

def beam_search_decoder(probs,labels, seq_len=None,  blankIdx = -1, beam_width=100):
    '''
    CTC beam search decoder

    Arguments:
        probs: The post-soft-max probabilities of the model, shape (batch_size,
            maximum sequence length, number of total labels + 1 ).
        labels: list of labels without the blank symbol
        seq_len: length of the sequence of each element in the batch,
            shape (bath_size)
        blankIdx: index of the blank symbol in the probs matrix
        beam_width: size of the beam
    Returns list of output label sequences with shape (batch_size)
    '''
    output_label = []
    for batch_idx in range(probs.shape[0]):
        prob_dist = probs[batch_idx][:seq_len[batch_idx]]

        beams = Beams()
        beams.new_beam(())
        beams.beams[()].prob_blank = 1.0
        beams.beams[()].prob_total = 1.0

        for t in range(prob_dist.shape[0]):
            top_beams = beams.get_top_beams()[:beam_width]
            new_beams = Beams()
            for labeling in top_beams:
                new_beams.new_beam(labeling)

                prob_non_blank = 0.0
                if labeling:
                    prob_non_blank = beams.beams[labeling].prob_non_blank * prob_dist[t][labeling[-1]]
                prob_blank = beams.beams[labeling].prob_total * prob_dist[t][blankIdx]
                new_beams.beams[labeling].prob_non_blank += prob_non_blank
                new_beams.beams[labeling].prob_blank += prob_blank
                new_beams.beams[labeling].prob_total += prob_non_blank + prob_blank

                for label_idx in range(len(labels)):
                    new_labeling = labeling + (label_idx,)
                    new_beams.new_beam(new_labeling)

                    if labeling and labeling[-1] == label_idx:
                        prob_non_blank = beams.beams[labeling].prob_blank * prob_dist[t][label_idx]
                    else:
                        prob_non_blank = beams.beams[labeling].prob_total * prob_dist[t][label_idx]

                    new_beams.beams[new_labeling].prob_non_blank += prob_non_blank
                    new_beams.beams[new_labeling].prob_total += prob_non_blank
            beams = new_beams
        output = ''
        for o in beams.get_top_beams()[0]:
            output += labels[o]
        output_label.append(output)
    return output_label

def best_path_decoder(probs,labels, seq_lens = None, blank_id=-1, allow_repetitions=False):
    '''
    CTC best path decoder

    Arguments:
        probs: The post-soft-max probabilities of the model, shape (batch_size,
            maximum sequence length, number of total labels + 1).
        labels: list of labels without the blank symbol
        seq_len: length of the sequence of each element in the batch,
            shape (bath_size)
        blankIdx: index of the blank symbol in the probs matrix
        allow_repetitions: if true, labels in the output symbol will be repeated

    Returns list of output label sequences with shape (batch_size)
    '''
    prob_choice = np.argmax(probs, axis=2)
    if blank_id == -1:
        blank_idx = len(labels)
    else:
        blank_idx = blank_id
    labels.insert(blank_idx, ' ')

    seq_list = []
    for i, row in enumerate(prob_choice):
        seq = []
        last_idx = -1
        for idx in row[:seq_lens[i]]:
            if idx == blank_idx:
                last_idx = idx
                continue
            if allow_repetitions:
                seq.append(labels[idx])
            else:
                if idx != last_idx or last_idx == -1:
                    last_idx = idx
                    seq.append(labels[idx])
        seq_list.append(''.join(seq))
    return seq_list

