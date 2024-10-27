import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from adjustText import adjust_text  # For adjusting text labels in plots
from matplotlib.lines import Line2D
from tqdm import tqdm
import logging
from matplotlib.colors import LinearSegmentedColormap
from math import comb
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define main concepts and their associated sub-concepts
CONCEPTS = {
    "Minimalism": ["simplicity", "reduction", "clarity", "clean lines", "functional purity"],
    "Innovation": ["creativity", "novelty", "cutting-edge", "experimentation", "new technology"],
    "Functionality": ["usefulness", "practicality", "efficiency", "multi-purpose", "ergonomics"],
    "Aesthetics": ["beauty", "visual appeal", "form", "balance", "proportion"],
    "Contextual": ["environment", "localism", "vernacular", "regionalism", "site-specific"],
    "Sustainability": ["eco-friendliness", "green building", "energy efficiency", "recycling", "low carbon"],
    "Tradition": ["historical reference", "cultural continuity", "heritage", "classical", "preservation"],
    "Avant-Garde": ["experimental", "progressive", "boundary-pushing", "revolutionary", "non-traditional"],
    "Universalism": ["global", "timeless", "cosmopolitan", "inclusive", "harmonious"],
    "Organic": ["integration with nature", "flowing forms", "natural materials", "holistic design", "biomimicry"]
}

# Load the SentenceTransformer model for generating text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_name):
    """
    Load data from a JSON file.

    Parameters:
    - file_name: Name of the JSON file to load.

    Returns:
    - Parsed JSON data as a Python object.
    """
    with open(file_name, 'r') as f:
        return json.load(f)

def verify_triplet_data(triplet_data_lookup):
    """
    Verify that all 10C3 triplet combinations are present in triplet_data_lookup.

    Parameters:
    - triplet_data_lookup: Dictionary storing triplet data.

    Returns:
    - None
    """
    expected_num_triplets = comb(10, 3)  # 10C3 = 120
    actual_num_triplets = len(triplet_data_lookup)
    if actual_num_triplets != expected_num_triplets:
        print(f"Warning: Expected {expected_num_triplets} triplets, but found {actual_num_triplets}.")
    else:
        print("All triplet combinations are present in triplet_data_lookup.")

    # Additionally, check for missing 'description' embeddings
    missing_descriptions = [
        triplet for triplet, data in triplet_data_lookup.items()
        if 'description' not in data.get('field_embeddings', {}) or
        data['field_embeddings']['description'] is None or
        data['field_embeddings']['description'].size == 0
    ]
    if missing_descriptions:
        print(f"Warning: {len(missing_descriptions)} triplets are missing 'description' embeddings.")
    else:
        print("All triplets have valid 'description' embeddings.")

def compute_main_concept_embeddings(sub_concept_embeddings):
    """
    Compute embeddings for main concepts by averaging their sub-concept embeddings.

    Parameters:
    - sub_concept_embeddings: Dictionary mapping sub-concept names to their embeddings.

    Returns:
    - main_concept_embeddings: Dictionary mapping main concept names to their embeddings.
    """
    main_concept_embeddings = {}
    for main_concept, sub_concepts in CONCEPTS.items():
        # Get embeddings of all sub-concepts for the main concept
        embeddings = [sub_concept_embeddings[sub_concept] for sub_concept in sub_concepts]
        # Compute the mean embedding for the main concept
        main_concept_embeddings[main_concept] = np.mean(embeddings, axis=0)
    return main_concept_embeddings

def compute_winner_embeddings(data, model, winner_data_lookup):
    """
    Compute embeddings for each winner by embedding each field separately and averaging.

    Parameters:
    - data: List of winner data dictionaries.
    - model: SentenceTransformer model for encoding text.
    - winner_data_lookup: Dictionary to store winner data and field embeddings.

    Returns:
    - Numpy array of winner embeddings.
    """
    winner_embeddings = []
    for winner in data:
        winner_fields_embeddings = {}
        winner_data_lookup[winner['name']] = {}
        for field_name, field_value in winner.items():
            if field_name == 'name':
                continue  # Skip the 'name' field
            if isinstance(field_value, str):
                # If the field is a string, encode it directly
                field_text = field_value
                winner_data_lookup[winner['name']][field_name] = field_text
                field_embedding = model.encode(field_text)
                winner_fields_embeddings[field_name] = field_embedding
            elif isinstance(field_value, list):
                # If the field is a list, join the elements and encode
                field_text = ' '.join(field_value)
                winner_data_lookup[winner['name']][field_name] = field_value
                field_embedding = model.encode(field_text)
                winner_fields_embeddings[field_name] = field_embedding
            elif isinstance(field_value, dict):
                # If the field is a dict, join the values and encode
                field_text = ' '.join([str(v) for v in field_value.values()])
                winner_data_lookup[winner['name']][field_name] = field_value
                field_embedding = model.encode(field_text)
                winner_fields_embeddings[field_name] = field_embedding
            else:
                continue  # Skip fields that are not str, list, or dict
        if winner_fields_embeddings:
            # Average embeddings of all fields to get the winner's embedding
            embeddings = list(winner_fields_embeddings.values())
            winner_embedding = np.mean(embeddings, axis=0)
            winner_embeddings.append(winner_embedding)
            winner_data_lookup[winner['name']]['field_embeddings'] = winner_fields_embeddings
        else:
            # If no fields, use a zero vector (embedding size is 384)
            winner_embeddings.append(np.zeros(384))
    return np.array(winner_embeddings)

def compute_triplet_embeddings(triplets_data, model, triplet_data_lookup):
    """
    Compute embeddings for each triplet by embedding each field separately and averaging.

    Parameters:
    - triplets_data: List of triplet data dictionaries.
    - model: SentenceTransformer model for encoding text.
    - triplet_data_lookup: Dictionary to store triplet data and field embeddings.

    Returns:
    - triplet_embeddings: Dictionary mapping triplet concepts set to their embeddings.
    """
    triplet_embeddings = {}
    for triplet in triplets_data:
        triplet_fields_embeddings = {}
        # Create a frozenset of concepts as the key to ensure order doesn't matter
        triplet_concepts_set = frozenset(triplet['concepts'])
        triplet_data_lookup[triplet_concepts_set] = {}
        for field_name, field_value in triplet.items():
            if field_name == 'concepts':
                continue  # Skip the 'concepts' field
            if isinstance(field_value, str):
                # If the field is a string, encode it directly
                field_text = field_value
                triplet_data_lookup[triplet_concepts_set][field_name] = field_text
                field_embedding = model.encode(field_text)
                triplet_fields_embeddings[field_name] = field_embedding
            elif isinstance(field_value, list):
                # If the field is a list, join the elements and encode
                field_text = ' '.join(field_value)
                triplet_data_lookup[triplet_concepts_set][field_name] = field_value
                field_embedding = model.encode(field_text)
                triplet_fields_embeddings[field_name] = field_embedding
            elif isinstance(field_value, dict):
                # If the field is a dict, join the values and encode
                field_text = ' '.join([str(v) for v in field_value.values()])
                triplet_data_lookup[triplet_concepts_set][field_name] = field_value
                field_embedding = model.encode(field_text)
                triplet_fields_embeddings[field_name] = field_embedding
            else:
                continue  # Skip fields that are not str, list, or dict
        if triplet_fields_embeddings:
            # Average embeddings of all fields to get the triplet's embedding
            embeddings = list(triplet_fields_embeddings.values())
            triplet_embedding = np.mean(embeddings, axis=0)
            triplet_embeddings[triplet_concepts_set] = triplet_embedding
            triplet_data_lookup[triplet_concepts_set]['field_embeddings'] = triplet_fields_embeddings
    return triplet_embeddings

def compute_similarities(winner_embeddings_norm, concept_embeddings_norm):
    """
    Compute cosine similarities between winners and concepts.

    Parameters:
    - winner_embeddings_norm: Normalized embeddings of winners.
    - concept_embeddings_norm: Normalized embeddings of concepts.

    Returns:
    - Similarity matrix of shape (num_winners, num_concepts).
    """
    return np.dot(winner_embeddings_norm, concept_embeddings_norm.T)

def select_top_unique_concepts(components, concept_names, concept_embeddings, num_components=3):
    """
    Select top concepts and match them with their semantic opposites, optimizing across components.

    Parameters:
    - components: Component loadings from PCA.
    - concept_names: List of main concept names.
    - concept_embeddings: Dictionary mapping concepts to embeddings.
    - num_components: Number of components to consider.

    Returns:
    - polar_opposites: List of tuples, each containing a concept and its semantic opposite.
    """
    polar_opposites = []
    used_concepts = set()

    for i in range(num_components):
        component = components[:, i]
        sorted_indices = np.argsort(component)
        concept_pos, concept_neg = None, None

        # Positive loading concepts
        for idx in reversed(sorted_indices):
            if component[idx] <= 0:
                break
            concept_candidate = concept_names[idx]
            if concept_candidate not in used_concepts:
                concept_pos = concept_candidate
                used_concepts.add(concept_pos)
                break

        # Negative loading concepts
        for idx in sorted_indices:
            if component[idx] >= 0:
                break
            concept_candidate = concept_names[idx]
            if concept_candidate not in used_concepts:
                concept_neg = concept_candidate
                used_concepts.add(concept_neg)
                break

        # If one of the concepts is missing, find the semantic opposite
        if concept_pos and not concept_neg:
            remaining_concepts = [c for c in concept_names if c not in used_concepts]
            if remaining_concepts:
                concept_neg = find_semantic_opposite(concept_pos, remaining_concepts, concept_embeddings)
                used_concepts.add(concept_neg)
        elif concept_neg and not concept_pos:
            remaining_concepts = [c for c in concept_names if c not in used_concepts]
            if remaining_concepts:
                concept_pos = find_semantic_opposite(concept_neg, remaining_concepts, concept_embeddings)
                used_concepts.add(concept_pos)

        if concept_pos and concept_neg:
            polar_opposites.append((concept_pos, concept_neg))

    return polar_opposites

def find_semantic_opposite(concept, candidates, concept_embeddings):
    """
    Find the most semantically opposite concept to a given concept from candidates.

    Parameters:
    - concept: The concept for which to find the opposite.
    - candidates: List of candidate concepts.
    - concept_embeddings: Dictionary mapping concepts to embeddings.

    Returns:
    - The concept from candidates that is most semantically opposite to the given concept.
    """
    concept_embedding = concept_embeddings[concept].reshape(1, -1)
    candidate_embeddings = [concept_embeddings[c] for c in candidates]
    similarities = cosine_similarity(concept_embedding, candidate_embeddings)[0]
    # The most opposite concept has the lowest similarity
    min_idx = np.argmin(similarities)
    return candidates[min_idx]

def determine_winner_hemisphere_and_influence(result_vectors, polar_opposites, data, winner_embeddings, analysis_type, main_concept_embeddings, model, winner_data_lookup):
    """
    Determine winner hemispheres and identify strongest influencing concepts.

    Parameters:
    - result_vectors: Array of transformed vectors (e.g., PCA projections).
    - polar_opposites: List of polar opposite pairs for each axis.
    - data: List of winner data dictionaries.
    - winner_embeddings: Array of winner embeddings.
    - analysis_type: String indicating the type of analysis (e.g., 'PCA').
    - main_concept_embeddings: Dictionary mapping main concepts to embeddings.
    - model: SentenceTransformer model.
    - winner_data_lookup: Dictionary storing winner data and field embeddings.

    Returns:
    - hemisphere_labels: List of hemisphere labels for each winner.
    - concept_scores: Array of concept scores for each winner.
    - concept_names_polar: List of main concept names.
    """
    hemisphere_labels, concept_scores = [], []
    concept_names_polar = list(main_concept_embeddings.keys())

    for idx, winner in enumerate(data):
        winner_name = winner['name']

        # Determine hemisphere by comparing results along each polar axis
        hemisphere_set = set()
        for axis in range(len(polar_opposites)):
            pos_concept, neg_concept = polar_opposites[axis]
            if pos_concept == "N/A" or neg_concept == "N/A":
                continue  # Skip if polar opposite is 'N/A'
            concept = pos_concept if result_vectors[idx, axis] >= 0 else neg_concept
            hemisphere_set.add(concept)

        if len(hemisphere_set) < 3:
            # Add additional concepts to make up the triplet
            remaining_concepts = [c for c in concept_names_polar if c not in hemisphere_set]
            hemisphere_set.update(remaining_concepts[:3 - len(hemisphere_set)])

        hemisphere = list(hemisphere_set)[:3]  # Ensure exactly 3 concepts
        hemisphere_label = ', '.join(hemisphere)
        hemisphere_labels.append(hemisphere_label)

        winner_embedding = winner_embeddings[idx]

        # Calculate similarities with all main concepts
        similarities = [
            cosine_similarity([winner_embedding], [main_concept_embeddings[concept]])[0][0]
            for concept in concept_names_polar
        ]
        concept_scores.append(similarities)

    return hemisphere_labels, np.array(concept_scores), concept_names_polar

def log_component_analysis(components, concept_names, analysis_type, polar_opposites, num_components=3):
    """
    Log the details of top components for PCA, including concept scores and inferred pairs.

    Parameters:
    - components: Component loadings from PCA.
    - concept_names: List of main concept names.
    - analysis_type: String indicating the type of analysis (e.g., 'PCA').
    - polar_opposites: List of polar opposite pairs for each axis.
    - num_components: Number of components to consider.
    """
    output_lines = []
    output_lines.append(f"\n{analysis_type} Top {num_components} Component Details:")
    for i in range(num_components):
        output_lines.append(f"\nComponent {i+1}:")
        component_scores = components[:, i]
        sorted_indices = np.argsort(component_scores)
        output_lines.append("Concept Scores:")
        for idx in reversed(sorted_indices):
            concept = concept_names[idx]
            score = component_scores[idx]
            output_lines.append(f"  {concept}: {score:.4f}")

        if i < len(polar_opposites):
            pos_concept, neg_concept = polar_opposites[i]
            pos_idx = concept_names.index(pos_concept)
            neg_idx = concept_names.index(neg_concept)
            output_lines.append(f"  Primary Concept ('{pos_concept}') Loading: {components[pos_idx, i]:.4f}")
            output_lines.append(f"  Opposite Concept ('{neg_concept}') Loading: {components[neg_idx, i]:.4f}")
            output_lines.append(f"  Inferred Polar Opposite Pair: '{pos_concept}' vs '{neg_concept}'")
        else:
            output_lines.append("  Inferred Polar Opposite Pair: 'N/A' vs 'N/A'")

    log_message = '\n'.join(output_lines)
    print(log_message)

def adjust_pca_signs(pca_components, concept_names, polar_opposites):
    """
    Adjust the signs of PCA components so that the primary concept has a positive loading.

    Parameters:
    - pca_components: PCA components array.
    - concept_names: List of main concept names.
    - polar_opposites: List of polar opposite pairs for each axis.

    Returns:
    - Adjusted PCA components.
    - Updated polar_opposites if signs were flipped.
    """
    adjusted_components = pca_components.copy()
    adjusted_polar_opposites = polar_opposites.copy()

    for i in range(len(polar_opposites)):
        pos_concept, neg_concept = polar_opposites[i]
        # Find indices of the positive and negative concepts in concept_names
        pos_idx = concept_names.index(pos_concept) if pos_concept in concept_names else None
        neg_idx = concept_names.index(neg_concept) if neg_concept in concept_names else None

        # Check the loading of the positive concept
        if pos_idx is not None and adjusted_components[pos_idx, i] < 0:
            # Flip the sign of the component
            adjusted_components[:, i] *= -1
            # Swap the polar opposites
            adjusted_polar_opposites[i] = (neg_concept, pos_concept)
            print(f"Flipped sign of Component {i+1} to make '{pos_concept}' positive.")
        else:
            print(f"Component {i+1} sign is correct for '{pos_concept}'.")

    return adjusted_components, adjusted_polar_opposites

def adjust_winner_positions_with_similarity_weights(
    winner_embeddings, winner_names, polar_opposites, triplets_data, winner_data_lookup,
    main_concept_embeddings, concept_embeddings_norm, triplet_data_lookup, sub_concept_embeddings,
    pca, scaled_similarities, scaler
):
    """
    Adjust winner embeddings based on positions of neighboring triplets in PCA space,
    using semantic similarities as weights.

    Parameters:
    - winner_embeddings: Original embeddings of winners.
    - winner_names: List of winner names.
    - polar_opposites: List of polar opposite pairs for each axis.
    - triplets_data: List of triplet data dictionaries.
    - winner_data_lookup: Dictionary storing winner data and field embeddings.
    - main_concept_embeddings: Dictionary mapping main concepts to embeddings.
    - concept_embeddings_norm: Normalized concept embeddings.
    - triplet_data_lookup: Dictionary storing triplet data and field embeddings.
    - sub_concept_embeddings: Dictionary mapping sub-concepts to embeddings.
    - pca: PCA model used in analysis.
    - scaled_similarities: Scaled similarity matrix before adjustment.
    - scaler: StandardScaler object used for scaling similarities.

    Returns:
    - adjusted_scaled_similarities: Adjusted scaled similarities of winners.
    - adjustment_records: Records of adjustments made for each winner.
    """
    adjusted_scaled_similarities = scaled_similarities.copy()
    adjustment_records = {}

    for idx, winner_name in enumerate(winner_names):
        winner_embedding = winner_embeddings[idx]

        # Check if winner_embedding is valid
        if np.linalg.norm(winner_embedding) == 0:
            print(f"Skipping winner {winner_name} due to zero embedding.")
            continue

        # Get winner's scaled similarities
        winner_scaled_similarity = scaled_similarities[idx]

        # Get winner's position in PCA space
        winner_coord = pca.transform(winner_scaled_similarity.reshape(1, -1))

        # Identify the current hemisphere (triplet)
        current_hemisphere = []
        for axis in range(len(polar_opposites)):
            pos_concept, neg_concept = polar_opposites[axis]
            if pos_concept == "N/A" or neg_concept == "N/A":
                continue  # Skip if polar opposite is 'N/A'
            concept = pos_concept if winner_scaled_similarity[axis] >= 0 else neg_concept
            current_hemisphere.append(concept)
        current_hemisphere_set = frozenset(current_hemisphere)  # Ensure frozenset for lookup

        # Ensure we have exactly 3 unique concepts
        if len(current_hemisphere_set) < 3:
            remaining_concepts = [c for c in main_concept_embeddings.keys() if c not in current_hemisphere_set]
            current_hemisphere.extend(remaining_concepts[:3 - len(current_hemisphere_set)])
            current_hemisphere_set = frozenset(current_hemisphere)

        # Get triplet description embedding for current hemisphere
        triplet_info_current = triplet_data_lookup.get(current_hemisphere_set)
        description_embedding = None
        if triplet_info_current and 'field_embeddings' in triplet_info_current:
            description_embedding = triplet_info_current['field_embeddings'].get('description')

        # Explicitly check if the description_embedding exists and is non-empty
        if description_embedding is not None and description_embedding.size > 0:
            # Compute similarities between triplet description embedding and main concept embeddings
            triplet_similarities_current = cosine_similarity([description_embedding], concept_embeddings_norm)[0]
            # Scale similarities
            scaled_triplet_similarities_current = scaler.transform(triplet_similarities_current.reshape(1, -1))
            # Map to PCA space
            coord_triplet_current = pca.transform(scaled_triplet_similarities_current)
        else:
            # If no description embedding, use winner's own coordinate
            coord_triplet_current = winner_coord

        # Compute similarity between winner embedding and current triplet description embedding
        if description_embedding is not None and description_embedding.size > 0:
            similarity_current = cosine_similarity([winner_embedding], [description_embedding])[0][0]
        else:
            similarity_current = 0

        # Initialize lists to store neighbor coordinates and similarities
        neighbor_coords = []
        similarities_neighbors = []
        for axis in range(len(polar_opposites)):
            pos_concept, neg_concept = polar_opposites[axis]
            if pos_concept == "N/A" or neg_concept == "N/A":
                continue  # Skip if polar opposite is 'N/A'

            # Flip the current axis value to the opposite in the hemisphere
            current_concept = current_hemisphere[axis] if axis < len(current_hemisphere) else pos_concept
            neighbor_concept = neg_concept if current_concept == pos_concept else pos_concept
            neighbor_hemisphere = list(current_hemisphere)
            if axis < len(neighbor_hemisphere):
                neighbor_hemisphere[axis] = neighbor_concept
            else:
                neighbor_hemisphere.append(neighbor_concept)
            neighbor_set = frozenset(neighbor_hemisphere)

            # Ensure we have exactly 3 unique concepts
            if len(neighbor_set) < 3:
                remaining_concepts = [c for c in main_concept_embeddings.keys() if c not in neighbor_set]
                neighbor_hemisphere.extend(remaining_concepts[:3 - len(neighbor_set)])
                neighbor_set = frozenset(neighbor_hemisphere)

            # Get triplet description embedding for neighbor hemisphere
            triplet_info_neighbor = triplet_data_lookup.get(neighbor_set)
            if triplet_info_neighbor and 'field_embeddings' in triplet_info_neighbor:
                triplet_desc_embedding_neighbor = triplet_info_neighbor['field_embeddings'].get('description')
                if triplet_desc_embedding_neighbor is not None and triplet_desc_embedding_neighbor.size > 0:
                    # Compute similarities between triplet description embedding and main concept embeddings
                    triplet_similarities_neighbor = cosine_similarity([triplet_desc_embedding_neighbor], concept_embeddings_norm)[0]
                    # Scale similarities
                    scaled_triplet_similarities_neighbor = scaler.transform(triplet_similarities_neighbor.reshape(1, -1))
                    # Map to PCA space
                    coord_triplet_neighbor = pca.transform(scaled_triplet_similarities_neighbor)
                    neighbor_coords.append(coord_triplet_neighbor[0])
                    # Compute similarity between winner embedding and neighbor triplet description embedding
                    similarity_neighbor = cosine_similarity([winner_embedding], [triplet_desc_embedding_neighbor])[0][0]
                    similarities_neighbors.append(similarity_neighbor)
                else:
                    # If no description embedding, use winner's own coordinate
                    neighbor_coords.append(winner_coord[0])
                    similarities_neighbors.append(0)
            else:
                # If triplet info missing, use winner's own coordinate
                neighbor_coords.append(winner_coord[0])
                similarities_neighbors.append(0)

        # Assign a base weight to the winner's own coordinate
        weight_winner = 1.0

        # Combine weights
        weights = [weight_winner, similarity_current] + similarities_neighbors

        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            # If all weights are zero, assign equal weights
            weights = np.ones_like(weights) / len(weights)

        # Compute the adjusted coordinate using weighted average
        all_coords = [winner_coord[0], coord_triplet_current[0]] + neighbor_coords
        adjusted_coord = np.average(all_coords, axis=0, weights=weights).reshape(1, -1)

        # Invert PCA transformation to get back to scaled similarities
        adjusted_scaled_similarity = pca.inverse_transform(adjusted_coord)

        # Check if adjusted_scaled_similarity is valid
        if np.isnan(adjusted_scaled_similarity).any() or np.isinf(adjusted_scaled_similarity).any():
            print(f"Adjusted scaled similarity contains NaN or Inf for winner {winner_name}")
            adjusted_scaled_similarity = winner_scaled_similarity.reshape(1, -1)

        # Update the winner's scaled similarities
        adjusted_scaled_similarities[idx] = adjusted_scaled_similarity[0]

        # Store adjustment details
        adjustment_records[winner_name] = {
            'hemisphere_before': current_hemisphere,
            'current_hemisphere_set': current_hemisphere_set,
            'delta_coord': adjusted_coord - winner_coord,
            'neighbors': neighbor_coords,
            'weights': weights,
            'similarity_current': similarity_current,
            'similarities_neighbors': similarities_neighbors
        }

    return adjusted_scaled_similarities, adjustment_records

def find_coinage_term_for_triplet(concepts_list, triplet_data_lookup):
    """
    Find the coinage term for a set of concepts regardless of order.

    Parameters:
    - concepts_list: List of concepts.
    - triplet_data_lookup: Dictionary storing triplet data.

    Returns:
    - Coinage term for the triplet, or 'Unknown' if not found.
    """
    concepts_set = frozenset(concepts_list)  # Ensure frozenset for lookup
    triplet = triplet_data_lookup.get(concepts_set)
    if triplet and 'coinage' in triplet and triplet['coinage']:
        return triplet['coinage']
    else:
        return 'Unknown'

def find_description_for_triplet(concepts_list, triplet_data_lookup):
    """
    Find the description for a set of concepts regardless of order.

    Parameters:
    - concepts_list: List of concepts.
    - triplet_data_lookup: Dictionary storing triplet data.

    Returns:
    - Description for the triplet, or 'No description available.' if not found.
    """
    concepts_set = frozenset(concepts_list)  # Ensure frozenset for lookup
    triplet = triplet_data_lookup.get(concepts_set)
    if triplet and 'description' in triplet and triplet['description']:
        return triplet['description']
    else:
        return 'No description available.'

def plot_3d_main_concepts(result, analysis_type, concept_names, polar_opposites):
    """
    Plot 3D visualization of main concept embeddings projected into PCA space.

    Parameters:
    - result: The PCA-transformed embeddings of main concepts.
    - analysis_type: Description of the analysis (e.g., '10 Main Concept Vectors').
    - concept_names: List of main concept names.
    - polar_opposites: List of polar opposite pairs for each axis.
    """
    # Normalize the result to have zero mean and unit variance
    scaler_plot = StandardScaler()
    result_norm = scaler_plot.fit_transform(result)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    offset = 0.05  # Adjust offset for label placement

    # Plot each point
    for i, name in enumerate(concept_names):
        ax.scatter(result_norm[i, 0], result_norm[i, 1], result_norm[i, 2], marker='o', s=60, alpha=0.8)
        ax.text(
            result_norm[i, 0] + offset, result_norm[i, 1] + offset, result_norm[i, 2] + offset,
            name, fontsize=9, ha='left', va='bottom'
        )

    # Set axis labels with appropriate padding
    ax.set_xlabel(f'{polar_opposites[0][0]} ← vs → {polar_opposites[0][1]}', fontsize=12, labelpad=15)
    # Flip labels for Component 2
    ax.set_ylabel(f'{polar_opposites[1][1]} ← vs → {polar_opposites[1][0]}', fontsize=12, labelpad=15)
    ax.set_zlabel(f'{polar_opposites[2][0]} ← vs → {polar_opposites[2][1]}', fontsize=12, labelpad=15)

    # Adjust the limits and aspect ratio
    max_range = np.array([
        result_norm[:, 0].max() - result_norm[:, 0].min(),
        result_norm[:, 1].max() - result_norm[:, 1].min(),
        result_norm[:, 2].max() - result_norm[:, 2].min()
    ]).max() / 2.0

    mid_x = (result_norm[:, 0].max() + result_norm[:, 0].min()) * 0.5
    mid_y = (result_norm[:, 1].max() + result_norm[:, 1].min()) * 0.5
    mid_z = (result_norm[:, 2].max() + result_norm[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=30)

    plt.title(f'3D Representation of Main Concepts ({analysis_type})', fontsize=16)

    plt.tight_layout()
    plt.show()
    
def calculate_main_concept_changes(scaled_similarities_before, scaled_similarities_after, scaler):
    """
    Calculate the changes in main concept similarities after adjustment.

    Parameters:
    - scaled_similarities_before: Scaled similarities before adjustment.
    - scaled_similarities_after: Scaled similarities after adjustment.
    - scaler: The StandardScaler object used for scaling.

    Returns:
    - changes: Array of changes in similarities.
    """
    similarities_before = scaler.inverse_transform(scaled_similarities_before)
    similarities_after = scaler.inverse_transform(scaled_similarities_after)
    changes = similarities_after - similarities_before
    return changes

def plot_3d_movements(result_before, result_after, analysis_type, laureate_names, hemisphere_labels_before, hemisphere_labels_after, triplet_data_lookup, polar_opposites):
    """
    Plot 3D movements of winners before and after adjustments.
    If a winner's position changes to a different hemisphere and coinage, their color updates accordingly.
    """
    # Combine result_before and result_after for consistent scaling
    combined_results = np.vstack((result_before, result_after))

    # Fit the scaler on the combined data
    scaler_movement = StandardScaler()
    combined_results_norm = scaler_movement.fit_transform(combined_results)

    # Split back into result_before_norm and result_after_norm
    result_before_norm = combined_results_norm[:result_before.shape[0], :]
    result_after_norm = combined_results_norm[result_before.shape[0]:, :]

    # Verify that both before and after results have zero mean and unit variance
    print("\nMovement Plot Normalization Check:")
    print("Result Before Mean:", np.mean(result_before_norm, axis=0))
    print("Result Before Std:", np.std(result_before_norm, axis=0))
    print("Result After Mean:", np.mean(result_after_norm, axis=0))
    print("Result After Std:", np.std(result_after_norm, axis=0))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Map hemisphere labels to coinage terms
    hemisphere_to_coinage_before = {
        h_label: find_coinage_term_for_triplet(h_label.split(', '), triplet_data_lookup)
        for h_label in set(hemisphere_labels_before)
    }
    hemisphere_to_coinage_after = {
        h_label: find_coinage_term_for_triplet(h_label.split(', '), triplet_data_lookup)
        for h_label in set(hemisphere_labels_after)
    }
    coinage_set = sorted(set(list(hemisphere_to_coinage_before.values()) + list(hemisphere_to_coinage_after.values())))
    hue_map = {term: idx / len(coinage_set) for idx, term in enumerate(coinage_set)}

    # Get hues for before and after positions
    hues_before = np.array([
        hue_map.get(hemisphere_to_coinage_before.get(hemisphere_labels_before[i], 'Unknown'), 0.0)
        for i in range(len(laureate_names))
    ])
    hues_after = np.array([
        hue_map.get(hemisphere_to_coinage_after.get(hemisphere_labels_after[i], 'Unknown'), 0.0)
        for i in range(len(laureate_names))
    ])

    offset = 0.05  # Adjust offset for label placement

    for i in range(len(laureate_names)):
        # Plot before and after positions with different markers and colors
        color_before = plt.cm.hsv(hues_before[i])
        color_after = plt.cm.hsv(hues_after[i])

        ax.scatter(result_before_norm[i, 0], result_before_norm[i, 1], result_before_norm[i, 2],
                   color=color_before, marker='o', edgecolors='k', s=60, alpha=0.8)

        ax.scatter(result_after_norm[i, 0], result_after_norm[i, 1], result_after_norm[i, 2],
                   color=color_after, marker='^', edgecolors='k', s=60, alpha=0.8)

        # Add an arrow to indicate movement
        ax.quiver(result_before_norm[i, 0], result_before_norm[i, 1], result_before_norm[i, 2],
                  result_after_norm[i, 0] - result_before_norm[i, 0],
                  result_after_norm[i, 1] - result_before_norm[i, 1],
                  result_after_norm[i, 2] - result_before_norm[i, 2],
                  color=color_after, arrow_length_ratio=0.0, linewidth=1, alpha=0.7)

        # Add the winner's last name at the final position
        ax.text(
            result_after_norm[i, 0] + offset, result_after_norm[i, 1] + offset, result_after_norm[i, 2] + offset,
            laureate_names[i].split()[-1], fontsize=9, ha='left', va='bottom'
        )

     # Set axis labels
    ax.set_xlabel(f'{polar_opposites[0][0]} ← vs → {polar_opposites[0][1]}', fontsize=12, labelpad=15)
    # Flip labels for Component 2
    ax.set_ylabel(f'{polar_opposites[1][1]} ← vs → {polar_opposites[1][0]}', fontsize=12, labelpad=15)
    ax.set_zlabel(f'{polar_opposites[2][0]} ← vs → {polar_opposites[2][1]}', fontsize=12, labelpad=15)

    # Adjust the limits and aspect ratio
    max_range = np.array([
        combined_results_norm[:, 0].max() - combined_results_norm[:, 0].min(),
        combined_results_norm[:, 1].max() - combined_results_norm[:, 1].min(),
        combined_results_norm[:, 2].max() - combined_results_norm[:, 2].min()
    ]).max() / 2.0

    mid_x = (combined_results_norm[:, 0].max() + combined_results_norm[:, 0].min()) * 0.5
    mid_y = (combined_results_norm[:, 1].max() + combined_results_norm[:, 1].min()) * 0.5
    mid_z = (combined_results_norm[:, 2].max() + combined_results_norm[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Create legend for coinage terms to the side with controlled spacing
    unique_terms = sorted(set(coinage_set))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'{term}',
               markerfacecolor=plt.cm.hsv(hue_map[term]),
               markeredgecolor='k', markersize=8) for term in unique_terms
    ]

    # Position legend outside the plot to the right
    ax.legend(handles=legend_elements, title='Coinage Terms', loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, title_fontsize=10)

    # Adjust viewing angle and save to file with consistent font settings
    ax.view_init(elev=20, azim=30)
    plt.title(f'3D Movements of Winners', fontsize=16)
    plt.show()

def main():
    """Main function orchestrating the analysis process."""
    # Load and preprocess data
    data = load_data('ppw.json')  # Replace with your data file
    data = [winner for winner in data if 'year' in winner and winner['year']]
    years = [int(winner['year']) for winner in data]
    laureate_names = [winner['name'] for winner in data]

    # Exclude 'Jimi Wen' if present
    if 'Jimi Wen' in laureate_names:
        idx = laureate_names.index('Jimi Wen')
        del data[idx]
        del years[idx]
        del laureate_names[idx]
        print("Excluded 'Jimi Wen' from the analysis.")

    print(f"Number of laureate names: {len(laureate_names)}")  # Should be 46

    # Embedding sub-concepts
    sub_concepts = [sub_concept for main_concept in CONCEPTS.values() for sub_concept in main_concept]
    sub_concept_embeddings = {sub_concept: model.encode(sub_concept) for sub_concept in sub_concepts}

    # Compute main concept embeddings
    main_concept_embeddings = compute_main_concept_embeddings(sub_concept_embeddings)
    concept_names = list(main_concept_embeddings.keys())
    concept_embeddings_matrix = np.array([main_concept_embeddings[concept] for concept in concept_names])

    # Normalize concept embeddings
    concept_embeddings_norm = normalize(concept_embeddings_matrix)

    # Compute winner embeddings
    winner_data_lookup = {}
    winner_embeddings = compute_winner_embeddings(data, model, winner_data_lookup)

    # Check for zero vectors in winner embeddings
    zero_vector_indices = [i for i, emb in enumerate(winner_embeddings) if np.linalg.norm(emb) == 0]
    if zero_vector_indices:
        print(f"Warning: Winner embeddings at indices {zero_vector_indices} are zero vectors.")
        print(f"Corresponding winners: {[laureate_names[i] for i in zero_vector_indices]}")
        # Remove winners with zero embeddings
        valid_indices = [i for i in range(len(winner_embeddings)) if np.linalg.norm(winner_embeddings[i]) != 0]
        winner_embeddings = winner_embeddings[valid_indices]
        laureate_names = [laureate_names[i] for i in valid_indices]
        years = [years[i] for i in valid_indices]
        data = [data[i] for i in valid_indices]
    else:
        print("All winner embeddings are valid.")

    # Normalize winner embeddings
    winner_embeddings_norm = normalize(winner_embeddings)

    # Compute similarities between winners and concepts
    similarities = compute_similarities(winner_embeddings_norm, concept_embeddings_norm)
    print("Similarities shape:", similarities.shape)
    print("Similarities sample values:", similarities[:5, :5])

    # Standardize similarities for PCA analysis
    scaler = StandardScaler()
    scaled_similarities = scaler.fit_transform(similarities)

    # Load triplet data and compute triplet embeddings
    triplets_data_full = load_data('artph.json')  # Replace with your data file
    triplets_data = triplets_data_full.get('triplets', [])
    triplet_data_lookup = {}
    triplet_embeddings = compute_triplet_embeddings(triplets_data, model, triplet_data_lookup)

    # Verify triplet data integrity
    verify_triplet_data(triplet_data_lookup)

    # Perform PCA analysis (First Pass) and log details
    pca_first = PCA(n_components=3)  # Limiting to 3 components for 3D visualization
    pca_result_first = pca_first.fit_transform(scaled_similarities)
    pca_components_first = pca_first.components_.T

    # Select polar opposites
    polar_opposites_pca = select_top_unique_concepts(pca_components_first, concept_names, main_concept_embeddings, num_components=3)
    log_component_analysis(pca_components_first, concept_names, "PCA (First Pass)", polar_opposites_pca, num_components=3)

    # Adjust PCA component signs
    pca_components_first_adjusted, polar_opposites_pca_adjusted = adjust_pca_signs(
        pca_components_first,
        concept_names,
        polar_opposites_pca
    )

    # Update PCA components in the PCA model
    pca_first.components_ = pca_components_first_adjusted.T

    # Re-transform data using adjusted PCA
    pca_result_first = scaled_similarities @ pca_first.components_.T

    # Normalize the PCA results to have zero mean and unit variance
    scaler_pca_first = StandardScaler()
    pca_result_first_norm = scaler_pca_first.fit_transform(pca_result_first)

    # Log component analysis with adjusted components
    log_component_analysis(pca_components_first_adjusted, concept_names, "PCA (After Sign Adjustment)", polar_opposites_pca_adjusted, num_components=3)

    # **First Pass Hemisphere Labels**
    # Determine hemisphere labels based on first pass PCA
    hemisphere_labels_pca_first, _, _ = determine_winner_hemisphere_and_influence(
        pca_result_first_norm,
        polar_opposites_pca_adjusted,
        data,
        winner_embeddings_norm,
        "PCA (First Pass)",
        main_concept_embeddings,
        model,
        winner_data_lookup
    )

    # Map hemisphere labels to coinage terms and descriptions for PCA First Pass
    hemisphere_to_coinage_pca_first = {}
    hemisphere_to_description_pca_first = {}
    for h_label in set(hemisphere_labels_pca_first):
        coinage_term = find_coinage_term_for_triplet(h_label.split(', '), triplet_data_lookup)
        description = find_description_for_triplet(h_label.split(', '), triplet_data_lookup)
        hemisphere_to_coinage_pca_first[h_label] = coinage_term
        hemisphere_to_description_pca_first[h_label] = description

    # Create hue maps for coloring
    def create_hue_map(hemisphere_to_coinage):
        unique_coinage = sorted(set(hemisphere_to_coinage.values()))
        hue_map = {term: idx / len(unique_coinage) for idx, term in enumerate(unique_coinage)}
        return hue_map

    hue_map_pca_first = create_hue_map(hemisphere_to_coinage_pca_first)

    # **Print Total Counts for Each Coinage (First Pass)**
    coinage_terms_pca_first = [hemisphere_to_coinage_pca_first[hemisphere_labels_pca_first[i]] for i in range(len(hemisphere_labels_pca_first))]
    coinage_counts_pca_first = Counter(coinage_terms_pca_first)
    print("\nCoinage counts for PCA First Pass:")
    for coinage, count in coinage_counts_pca_first.items():
        print(f"{coinage}: {count}")

    # **Option 1: Adjust Winner Positions Using Similarity Weights**
    adjusted_scaled_similarities_pca_opt1, adjustment_records_pca_opt1 = adjust_winner_positions_with_similarity_weights(
        winner_embeddings, laureate_names, polar_opposites_pca_adjusted, triplets_data,
        winner_data_lookup, main_concept_embeddings, concept_embeddings_norm, triplet_data_lookup, sub_concept_embeddings,
        pca_first, scaled_similarities, scaler
    )

    # For PCA Option 1
    pca_result_second_pca_opt1 = pca_first.transform(adjusted_scaled_similarities_pca_opt1)

    # **Compute Absolute Adjustments for Each Winner**
    adjustments = np.linalg.norm(adjusted_scaled_similarities_pca_opt1 - scaled_similarities, axis=1)

    # **Normalize the adjusted PCA results**
    scaler_pca_second = StandardScaler()
    pca_result_second_norm_pca_opt1 = scaler_pca_second.fit_transform(pca_result_second_pca_opt1)

    # **Determine Hemisphere Labels Using Adjusted Positions**
    # For PCA Option 1
    hemisphere_labels_pca_opt1, _, _ = determine_winner_hemisphere_and_influence(
        pca_result_second_norm_pca_opt1,
        polar_opposites_pca_adjusted,
        data,
        winner_embeddings_norm,
        "PCA Option 1",
        main_concept_embeddings,
        model,
        winner_data_lookup
    )

    # Map hemisphere labels to coinage terms and descriptions for PCA Option 1
    hemisphere_to_coinage_pca_opt1 = {}
    hemisphere_to_description_pca_opt1 = {}
    for h_label in set(hemisphere_labels_pca_opt1):
        coinage_term = find_coinage_term_for_triplet(h_label.split(', '), triplet_data_lookup)
        description = find_description_for_triplet(h_label.split(', '), triplet_data_lookup)
        hemisphere_to_coinage_pca_opt1[h_label] = coinage_term
        hemisphere_to_description_pca_opt1[h_label] = description

    hue_map_pca_opt1 = create_hue_map(hemisphere_to_coinage_pca_opt1)

    # **Print Total Counts for Each Coinage (After Adjustments)**
    # For PCA Option 1
    coinage_terms_pca_opt1 = [hemisphere_to_coinage_pca_opt1[hemisphere_labels_pca_opt1[i]] for i in range(len(hemisphere_labels_pca_opt1))]
    coinage_counts_pca_opt1 = Counter(coinage_terms_pca_opt1)
    print("\nCoinage counts for PCA Option 1:")
    for coinage, count in coinage_counts_pca_opt1.items():
        print(f"{coinage}: {count}")

    # **Identify Winners Who Changed Hemisphere**
    winners_changed_hemisphere = [
        idx for idx in range(len(laureate_names))
        if hemisphere_labels_pca_first[idx] != hemisphere_labels_pca_opt1[idx]
    ]

    if len(winners_changed_hemisphere) == 0:
        print("\nNo winners changed hemispheres after adjustment.")
    else:
        # **Get Adjustments for Winners Who Changed Hemisphere**
        adjustments_changed = adjustments[winners_changed_hemisphere]
        laureate_names_changed = [laureate_names[idx] for idx in winners_changed_hemisphere]
        data_changed = [data[idx] for idx in winners_changed_hemisphere]
        hemisphere_labels_before_changed = [hemisphere_labels_pca_first[idx] for idx in winners_changed_hemisphere]
        hemisphere_labels_after_changed = [hemisphere_labels_pca_opt1[idx] for idx in winners_changed_hemisphere]

        # **Select Top 3 Winners with Largest Adjustments**
        if len(adjustments_changed) >= 8:
            top_3_indices = np.argsort(-adjustments_changed)[:8]  # Indices of top 3 largest adjustments
        else:
            top_3_indices = np.argsort(-adjustments_changed)  # If fewer than 3 winners changed hemisphere

        print("\nTop 3 Winners with Largest Absolute Adjustments (among those who changed hemisphere):")

        for idx in top_3_indices:
            winner_name = laureate_names_changed[idx]
            adjustment_value = adjustments_changed[idx]
            winner_data = data_changed[idx]
            before_hemisphere = hemisphere_labels_before_changed[idx]
            after_hemisphere = hemisphere_labels_pca_opt1[idx]

            # Get winner's fields and content
            fields_and_content = winner_data_lookup[winner_name]

            # Get before and after hemisphere coinage terms and descriptions
            before_coinage = hemisphere_to_coinage_pca_first.get(before_hemisphere, 'Unknown')
            after_coinage = hemisphere_to_coinage_pca_opt1.get(after_hemisphere, 'Unknown')
            before_description = hemisphere_to_description_pca_first.get(before_hemisphere, 'No description available.')
            after_description = hemisphere_to_description_pca_opt1.get(after_hemisphere, 'No description available.')

            print(f"\nWinner: {winner_name}")
            print(f"  Absolute Adjustment: {adjustment_value:.4f}")
            print(f"  Hemisphere Before Adjustment: {before_hemisphere} (Coinage: {before_coinage})")
            print(f"    Description: {before_description}")
            print(f"  Hemisphere After Adjustment: {after_hemisphere} (Coinage: {after_coinage})")
            print(f"    Description: {after_description}")
            print("  Fields and Content:")
            for field, content in fields_and_content.items():
                if field != 'field_embeddings':
                    print(f"    {field}: {content}")

    # **Plot 3D Movements for PCA Option 1**
    # Optionally, plot 3D movements if needed
    plot_3d_movements(
        pca_result_first_norm[:, :3],
        pca_result_second_norm_pca_opt1[:, :3],
        "PCA Option 1",
        laureate_names,
        hemisphere_labels_pca_first,    # Before adjustment
        hemisphere_labels_pca_opt1,     # After adjustment
        triplet_data_lookup,
        polar_opposites_pca_adjusted
    )

    # **Convert Normalized Positions Back to Main Concepts Space**
    # Invert PCA transformation to get back to scaled similarities
    adjusted_scaled_similarities_pca_opt1 = pca_first.inverse_transform(pca_result_second_pca_opt1)

    # **Compute Main Concept Changes Based on Adjusted Positions**
    main_concept_changes = calculate_main_concept_changes(scaled_similarities, adjusted_scaled_similarities_pca_opt1, scaler)
    print("Main concept changes shape:", main_concept_changes.shape)
    print("Main concept changes sample values:", main_concept_changes[:5, :5])

    # Map hemisphere labels to coinage terms per winner
    coinage_terms_before = [hemisphere_to_coinage_pca_first[hemisphere_labels_pca_first[i]] for i in range(len(laureate_names))]
    coinage_terms_after = [hemisphere_to_coinage_pca_opt1[hemisphere_labels_pca_opt1[i]] for i in range(len(laureate_names))]

    # **Extract Similarities Before and After Adjustment**
    # Unscaled similarities (before adjustment)
    similarities_before = scaler.inverse_transform(scaled_similarities)
    # Unscaled similarities (after adjustment)
    similarities_after = scaler.inverse_transform(adjusted_scaled_similarities_pca_opt1)
    # Difference between after and before
    differences = similarities_after - similarities_before

    # **Create DataFrames for Better Visualization**
    df_before = pd.DataFrame(similarities_before, index=laureate_names, columns=concept_names)
    df_after = pd.DataFrame(similarities_after, index=laureate_names, columns=concept_names)
    df_difference = pd.DataFrame(differences, index=laureate_names, columns=concept_names)

    # **Print the Matrices**
    print("\nMain Concept Scores Before Adjustment:")
    print(df_before)

    print("\nMain Concept Scores After Adjustment:")
    print(df_after)

    print("\nDifference (After - Before):")
    print(df_difference)

    # **Plot the Main Concepts in PCA Space**
    # Extract embeddings into a NumPy array
    embeddings_array = np.array([main_concept_embeddings[concept] for concept in concept_names])

    # Normalize main concept embeddings
    main_concept_embeddings_norm = normalize(embeddings_array)

    # Compute similarities between main concepts and main concepts
    similarities_main_concepts = compute_similarities(main_concept_embeddings_norm, concept_embeddings_norm)

    # Standardize similarities using the same scaler
    scaled_similarities_main_concepts = scaler.transform(similarities_main_concepts)

    # Project into PCA space using the same PCA model
    pca_result_main_concepts = pca_first.transform(scaled_similarities_main_concepts)

    # Normalize the PCA results
    pca_result_main_concepts_norm = scaler_pca_first.transform(pca_result_main_concepts)

    # **Plot the Main Concepts**
    plot_3d_main_concepts(
        pca_result_main_concepts_norm,
        "10 Main Concept Vectors",
        concept_names,
        polar_opposites_pca_adjusted
    )

if __name__ == "__main__":
    main()
