# Sequence (re)-generation with different orderings

To run experiments on the MultiNLI dataset, use the file `mnli_regen.py`. By default, the code performs likelihood-based masking. To change this, alter the `setting` parameter in the file. It can take on the values `random`, `likelihood`, or `likelihood_plus_exploration` corresponding to the different masking strategies specified. The output generated upon running this will look like the below.

ORIG: [original sequence]

MASK1: [original masked sequence]

RGEN1: [1st generated sequence]

MASK2: [2nd re-masked sequence]

RGEN2: [2nd (re)-generated sequence]

MASK3: [3rd masked sequence]

RGEN3: [3rd (re)-generated sequence]

The argument `show_mask=True` in the `print_regen` function allows the masked and re-masked sequences to be printed as above. To only print the set of generated sequences, set `show_mask=False`.

