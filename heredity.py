import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)
                
    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")

    # For debugging
    # print(probabilities)


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_genes` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Compute prob that everyone in set `one_gene` has one copy of the gene
    P_one_gene = oneGeneProb(people, one_gene, two_genes)

    # Compute prob that everyone in set `two_genes` has two copies of the gene
    P_two_genes = twoGenesProb(people, one_gene, two_genes)

    # Compute prob that everyone not in `one_gene` or `two_genes` does not have the gene
    P_not_genes = notGenesProb(people, one_gene, two_genes)

    # Compute prob that everyone in set `have_trait` has the trait
    P_trait = haveTraitProb(people, one_gene, two_genes, have_trait)

    # Compute prob that everyone not in set` have_trait` does not have the trait
    P_not_trait = notTraitProb(people, one_gene, two_genes, have_trait)

    P_joint = P_one_gene*P_two_genes*P_not_genes*P_trait*P_not_trait

    
    return P_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        # Update gene probability
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        # Update trait probability
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Iterate through all people
    for person in probabilities:

        # Get gene values
        Pgene0 = probabilities[person]["gene"][0]
        Pgene1 = probabilities[person]["gene"][1]
        Pgene2 = probabilities[person]["gene"][2]

        # Calculate normalized gene values and update them
        factor = Pgene0 + Pgene1 + Pgene2
        if factor != 0:
            probabilities[person]["gene"][0] = Pgene0/factor
            probabilities[person]["gene"][1] = Pgene1/factor
            probabilities[person]["gene"][2] = Pgene2/factor

        # Get trait values
        PtraitTrue = probabilities[person]["trait"][True]
        PtraitFalse = probabilities[person]["trait"][False]

        # Calculate normalized trait values and update them
        factor = PtraitTrue + PtraitFalse
        if factor != 0:
            probabilities[person]["trait"][True] = PtraitTrue/factor
            probabilities[person]["trait"][False] = PtraitFalse/factor


def childNoGeneProb(people, one_gene, two_genes, person):
    """
    Computes the probability of the child NOT getting the gene, which is:
        - prob that he does not get the gene from his mother and he does not get the gene from his father (Pcase)
    """
    # In this case, the probability is the product of:
    # - prob if mother has 0 (Pmom0), 1 (Pmom1) or 2 (Pmom2) and it is not passed on (part1)
    if people[person]["mother"] in one_gene:
        part1 = 0.5
    elif people[person]["mother"] in two_genes:
        part1 = PROBS["mutation"]
    else:
        part1 = 1-PROBS["mutation"]

    # - prob if father has 0 (Pdad0), 1 (Pdad1) or 2 (Pdad2) and it is not passed on (part2)
    if people[person]["father"] in one_gene:
        part2 = 0.5
    elif people[person]["father"] in two_genes:
        part2 = PROBS["mutation"]
    else:
        part2 = 1-PROBS["mutation"]

    Pcase = part1 * part2
    # print(Pcase)

    return Pcase


def childOneGeneProb(people, one_gene, two_genes, person):
    """
    Computes the probability of the child getting one gene, which is the sum of:
        - prob that he gets one gene from his mother and not his father (Pcase1),
        - prob that he gets one gene from his father and not his mother (Pcase2).
    """
    # In case 1, the probability is the product of:
    # - prob if mother has 0 (Pmom0), 1 (Pmom1) or 2 (Pmom2) and it is passed on (part1)
    if people[person]["mother"] in one_gene:
        part1 = 0.5
    elif people[person]["mother"] in two_genes:
        part1 = 1 - PROBS["mutation"]
    else:
        part1 = PROBS["mutation"]

    # - prob if father has 0 (Pdad0), 1 (Pdad1) or 2 (Pdad2) and it is not passed on (part2)
    if people[person]["father"] in one_gene:
        part2 = 0.5
    elif people[person]["father"] in two_genes:
        part2 = PROBS["mutation"]
    else:
        part2 = 1-PROBS["mutation"]

    Pcase1 = part1 * part2

    # In case 2, the probability is the product of:
    # - prob if father has 0 (Pmom0), 1 (Pmom1) or 2 (Pmom2) and it is passed on (part1)
    if people[person]["father"] in one_gene:
        part1 = 0.5
    elif people[person]["father"] in two_genes:
        part1 = 1 - PROBS["mutation"]
    else:
        part1 = PROBS["mutation"]

    # - prob if mother has 0 (Pdad0), 1 (Pdad1) or 2 (Pdad2) and it is not passed on (part2)
    if people[person]["mother"] in one_gene:
        part2 = 0.5
    elif people[person]["mother"] in two_genes:
        part2 = PROBS["mutation"]
    else:
        part2 = 1-PROBS["mutation"]

    Pcase2 = part1 * part2

    # print(Pcase1+Pcase2)

    # The probability for the child to get one gene is the product of the two cases
    return Pcase1+Pcase2


def childTwoGenesProb(people, one_gene, two_genes, person):
    """
    Computes the probability of the child getting two genes, which is:
        - prob that he gets one gene from his mother and one from his father (Pcase)
    """
    # In case 1, the probability is the product of:
    # - prob if mother has 0 (Pmom0), 1 (Pmom1) or 2 (Pmom2) and it is passed on (part1)
    if people[person]["mother"] in one_gene:
        part1 = 0.5
    elif people[person]["mother"] in two_genes:
        part1 = 1 - PROBS["mutation"]
    else:
        part1 = PROBS["mutation"]

    # - prob if father has 0 (Pdad0), 1 (Pdad1) or 2 (Pdad2) and it is passed on (part2)
    if people[person]["father"] in one_gene:
        part2 = 0.5
    elif people[person]["father"] in two_genes:
        part2 = 1 - PROBS["mutation"]
    else:
        part2 = PROBS["mutation"]
    
    Pcase = part1 * part2
    # print(Pcase)

    return Pcase


def oneGeneProb(people, one_gene, two_genes):
    """
    Compute and return the probability that everyone in set `one_gene` has one copy of the gene.
    """
    # Compute prob that each one in set "one_gene" has one copy of gene and make a list
    P_one_gene_list = []
    for person in one_gene:
        
        # If the person is a parent, use general PROBS
        if people[person]["father"] == None:
            P_one_gene_list.append(PROBS["gene"][1])

        # Else, if person is a child:
        else:
            prob = childOneGeneProb(people, one_gene, two_genes, person)
            P_one_gene_list.append(prob)
    
    # Compute joint probability of list (product)
    P_one_gene = 1
    for i in range(len(P_one_gene_list)):
        P_one_gene = P_one_gene_list[i]*P_one_gene
    
    return P_one_gene


def twoGenesProb(people, one_gene, two_genes):
    """
    Compute and return the probability that everyone in set `two_genes` has two copies of the gene.
    """
    # Compute prob that each one in set "two_genes" has two copies of gene and make a list
    P_two_genes_list = []
    for person in two_genes:
        
        # If the person is a parent, use general PROBS
        if people[person]["father"] == None:
            P_two_genes_list.append(PROBS["gene"][2])

        # Else, if person is a child:
        else:
            prob = childTwoGenesProb(people, one_gene, two_genes, person)
            P_two_genes_list.append(prob)
    
    # Compute joint probability of list (product)
    P_two_genes = 1
    for i in range(len(P_two_genes_list)):
        P_two_genes = P_two_genes_list[i]*P_two_genes
    
    return P_two_genes


def notGenesProb(people, one_gene, two_genes):
    """
    Compute and return the probability that everyone not in `one_gene` or `two_genes` does not have the gene.
    """
    # Compute prob that each one not in `one_gene` or `two_genes` does not have the gene and make a list
    P_not_genes_list = []
    for person in people:

        # Check that the person is not in one_gene or two_genes
        if person not in one_gene and person not in two_genes:
        
            # If the person is a parent, use general PROBS
            if people[person]["father"] == None:
                P_not_genes_list.append(PROBS["gene"][0])

            # Else, if person is a child:
            else:
                prob = childNoGeneProb(people, one_gene, two_genes, person)
                P_not_genes_list.append(prob)
    
    # Compute joint probability of list (product)
    P_not_genes = 1
    for i in range(len(P_not_genes_list)):
        P_not_genes = P_not_genes_list[i]*P_not_genes

    return P_not_genes


def haveTraitProb(people, one_gene, two_genes, have_trait):
    """
    Compute and return the probability that everyone in set `have_trait` has the trait.
    """
    # Compute prob that each one in `have_trait` has the trait and make a list
    P_trait_list = []
    for person in have_trait:
        
        if person in one_gene:
            prob = PROBS["trait"][1][True]
        elif person in two_genes:
            prob = PROBS["trait"][2][True]
        else:
            prob = PROBS["trait"][0][True]
        P_trait_list.append(prob)
        
    # Compute joint probability of list (product)
    P_trait = 1
    for i in range(len(P_trait_list)):
        P_trait = P_trait_list[i]*P_trait

    return P_trait


def notTraitProb(people, one_gene, two_genes, have_trait):
    """
    Compute and return the probability that everyone not in set` have_trait` does not have the trait.
    """
    # Compute prob that each one not in `have_trait` does not have the gene and make a list
    P_not_trait_list = []
    for person in people:

        # Check that the person is not in have_trait
        if person not in have_trait:

            if person in one_gene:
                prob = PROBS["trait"][1][False]
            elif person in two_genes:
                prob = PROBS["trait"][2][False]
            else:
                prob = PROBS["trait"][0][False]
            P_not_trait_list.append(prob)
        
    # Compute joint probability of list (product)
    P_not_trait = 1
    for i in range(len(P_not_trait_list)):
        P_not_trait = P_not_trait_list[i]*P_not_trait

    return P_not_trait


if __name__ == "__main__":
    main()
