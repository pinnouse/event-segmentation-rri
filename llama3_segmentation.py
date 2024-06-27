import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, zscore
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import re
from typing import List

import torch
from transformers import pipeline

pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")


# Define function to remove quotations and punctuations from text
def clean_text(text_path):
	""" Given a text path, return a string of text that has quotation marks, ellipses, and dashes removed.

	Argument:
	text_path - path name of story text file

	Return:
	text - cleaned story text
	"""

	text_file = open(text_path, 'r', encoding='utf-8-sig')
	text = text_file.read()
	text_file.close()

	text = text.replace('"', '')
	text = text.replace('\u201C', '')
	text = text.replace('\u201D', '')
	# text = text.replace('\u2018', '') # Single quotation
	# text = text.replace('\u2019', '')
	text = text.replace('...', '.')
	text = text.replace('-', ' ')

	return text


# Parse Segmentation LLM outputs
def parse_llm_output(output: str):
	# Correct all the new line characters: i.e. convert "\n \n" into "\n\n"
	temp_output = re.sub('(?<=\\n)\\s(?=\\n)', '\\n\\n', output)

	# Replace all the \n to \n\n
	temp_output = re.sub('(?<!\\n)\\n(?!\\n)', '\\n\\n', temp_output)

	# Remove all the numbers
	temp_output = re.sub(r"[0-9]+\.", "", temp_output)

	# Remove all the Event #: into empty string
	temp_output = re.sub(r"Event \d+:", "", temp_output)

	# Remove all the Segment #\n: into empty string
	temp_output = re.sub(r"Segment \d+: ", "", temp_output)

	# Replace all the space after the \n\n
	temp_output = re.sub("(?<=\\n\\n)\\s", "", temp_output)

	# Replace all the space before the \n\n
	temp_output = re.sub(" (?=\\n\\n)", "", temp_output)

	# Remove the space in the beginning
	temp_output = re.sub("^\\s", "", temp_output)

	# Remove the new line in the beginning
	# temp_output = re.sub("^\n", "", temp_output)
	temp_output = temp_output.lstrip()

	return temp_output


# Define Prompt Function
def prompt(message, temperature=0, max_tokens=4096, frequency_penalty=0):
	"""Prompt the LLM model

	Args:
		message: The message send to the language model
		temperature: Optional parameter that changes the randomness of the
			language model. Default is 0
		top_P: Optional parameter that changes the threshold when nucleus
			sampling. Default is 1
		max_tokens: Optional parameter that changes the max number of tokens
			that language model can generate. Default is 4096
		frequency_penalty: Optional parameter that changes the behaviour of the
			language model. Default value is 0 i.e. positive value discourages
			the model from generating commonly used words and negative value
			encourage the model to stick to more common words or phrases.


	Returns:
		A chat completion response. Example response would look like the
		following:

		[{
			"generated_text": [
				{ "role": "assistant", "content": "..." }
			]
		}, ...]
	"""
	prompt_message = [{"role": "user", "content": message}]
	response = pipe(prompt_message,
				 max_new_tokens=max_tokens,
				 temperature=temperature,
				 repetition_penalty=(1+frequency_penalty))

	return response


def get_output(responses, choice: int = 0):
	"""Read content form LLM chat completion.

	Args:
		responses: the OpenAIObject chat.Completion
		choice: the choice of generated message
	"""
	return responses[choice]['generated_text'][-1].content


def split_text(parsed_text: List[str]) -> List[list]:
	""" Given the parsed_text from gpt_segmentation, separate the segmentation into items in a list.

	:param parsed_text: list of strings containing segmented outputs
	:return: list containing individual events as items
	"""

	segmented_events = []
	for i in parsed_text:
		segmented_events.append(i.split('\n\n'))

	return segmented_events


def llm_segmentation(text_path, iters=1):
	""" Given a text file, GPT model, and number of iterations, output the GPT segmented responses.

	[tt, hh] = gpt_segmentation(text_path, iters, model):

	Arguments:
	text -- story as a text file
	model -- The language model to prompt
	iteration -- number of time for the story to be segmented

	Returns:
	parsedLLM - list of segmented events
	"""

	text = clean_text(text_path)

	prompt_onset = "An event is an ongoing coherent situation. The following story needs to be copied and segmented into \
	large events. Copy the following story word-for-word and start a new line whenever one event ends and another begins. \
	This is the story: "

	prompt_offset = "\n This is a word-for-word copy of the same story that is segmented into large event units: "

	responses = []
	iter_times = iters

	for i in tqdm.tqdm(range(iter_times)):
		curr_prompt = prompt_onset + text + prompt_offset

		curr_response = prompt(curr_prompt)

		# Save the current model
		responses.append(curr_response)

	parsed_LLM = [parse_llm_output(get_output(item)) for item in responses]

	segmented_events = split_text(parsed_LLM)

	return segmented_events


def find_event_boundaries(word_num: List[int]) -> List[int]:
	"""Given a list of word numbers representing the lengths of events, 
    return a list of integers with the associated word numbers at event boundaries.

    Args:
        word_num (list[int]): List of word numbers associated with events.

    Returns:
        list[int]: List of word numbers at event boundary locations.
	"""
	boundaries = []

	for i in range(1, len(word_num)):
		# Cumulative word count up to the current event and 
		# add 1 because event boundary is located at successive word
		word_count = sum(word_num[:i]) + 1  
		if word_count != sum(word_num[:i - 1]):
			boundaries.append(word_count)

	return boundaries

def event_data(events: list[str]) -> list[int]:
	""" Given a list of events, return a list of integers with the associated 
	word numbers.

	Args:
		events (list[str]): List of events from gpt_segmentation output

	Return:
		list[int]: List of word numbers associated with event boundary location
	"""

	length_events = [len(event.split()) for event in events]
	word_boundaries = find_event_boundaries(length_events)

	return word_boundaries


def batch_emb(model, sentences: list) -> list:
	"""Convert a batch of sentences into embedding

	Keyword arguments:
	model -- The model used for encoding
	sentences -- A list of sentences
	"""

	emb_sentences = []
	for sentence in sentences:
		emb_sentences.append(model(sentence))

	return emb_sentences


def event_centrality(embeddings: list, correlation=spearmanr, fisherz=False, plot=False, z_score=None) -> np.ndarray:
	""" Given a list of sentence embeddings, output a matrix of correlation values
	to identify the centrality of each event.

	Args:
		embeddings (list): Sentence embedding vectors
		correlation (function): Correlation function from scipy.stats to use to calculate the matrix
								(default is spearmanr)
		fisherz (bool): Fishers Z-Transformation of marix (default is False)
		plot (bool): Show the matrix as an annotated plot (default is False)

	Returns:
		np.array: Array of correlation values representing event centrality

	"""

	num_events = len(embeddings)
	matrix = np.zeros((num_events, num_events))

	for i in range(num_events):
		for j in range(num_events):
			matrix[i][j] = correlation(embeddings[i], embeddings[j]).statistic

	if fisherz:
		matrix = np.arctanh(matrix)

	if plot:
		ax = sns.clustermap(matrix, col_cluster=False, row_cluster=False, annot=True, z_score=z_score, cmap='Blues')

		ax.ax_heatmap.set_ylabel('Events')
		ax.ax_heatmap.set_xlabel('Events')
		plt.show()

	return matrix


def recall_matrix(story_embedding: list, recall_embedding: list, correlation=spearmanr, fisherz=False, plot: object = False,
				  z_score=False) -> np.ndarray:
	""" Given a list of sentence embeddings for the narrative and recall transcripts, output a matrix of correlation
	values to identify which narrative events were more strongly recalled.

	Args:
		story_embedding (list): Sentence embedding vectors for story events
		recall_embedding (list): Sentence embedding vectors for recall events
		correlation (function): Correlation function from scipy.stats to use to calculate the matrix
								(default is spearmanr)
		fisherz (bool): Fishers Z-Transformation of Matrix (defaults is False)
		plot (bool): Show the matrix as an annotated plot (default is False)
		z_score (bool): Z-Score across rows of matrix (default is False)

	Returns:
		np.ndarray: Array of correlation values representing event recall
	"""

	num_story_events = len(story_embedding)
	num_recall_events = len(recall_embedding)

	matrix = np.zeros((num_story_events, num_recall_events))

	for i in range(num_story_events):
		for j in range(num_recall_events):

			if correlation == spearmanr or correlation == pearsonr:
				matrix[i][j] = correlation(story_embedding[i], recall_embedding[j]).statistic

			elif correlation == np.linalg.norm:
				matrix[i][j] = correlation(story_embedding[i] - recall_embedding[j])
				
			else:
				matrix[i][j] = correlation(story_embedding[i], recall_embedding[j])

	# Fisher Z-transformation
	if fisherz:
		matrix = np.arctanh(matrix)

	if z_score:
		matrix = zscore(matrix, axis=None)

	if plot and z_score:
		ax = sns.clustermap(matrix, col_cluster=False, row_cluster=False, annot=True, z_score=1, cmap='Blues')

		ax.ax_heatmap.set_ylabel('Story Events')
		ax.ax_heatmap.set_xlabel('Recall Events')
		plt.show()

	elif plot and not z_score:
		ax = sns.clustermap(matrix, col_cluster=False, row_cluster=False, annot=True, z_score=None, cmap='Blues')

		ax.ax_heatmap.set_ylabel('Story Events')
		ax.ax_heatmap.set_xlabel('Recall Events')
		plt.show()

	return matrix


def recall_score(matrix: np.ndarray) -> int:
	"""Given a matrix of standardized correlation values with the rows representing narrative events and columns
	representing recall events, return a recall score.

	Args:
		matrix (np.ndarray): Standardized correlation matrix

	Returns:
		int: Recall Z-Score
	"""

	score = np.mean(np.max(matrix, axis=1))

	return score


def actual_recall_score(story_embedding: list, recall_embedding: list, correlation=spearmanr, z_score=False) -> int:
	""" Given a vector of sentence of embeddings for corresponding stories, calculate the actually recall score with
	the actuall recall embedding.

	Args:
		story_embedding (list): Sentence embedding vector for corresponding story
		recall_embedding (list): Sentence embedding vectors for recall events
		z_score (bool): z-score recall matrix (default is False)

	Returns:
		int: Actual Recall Score
	"""

	score = recall_score(recall_matrix(story_embedding, recall_embedding, correlation, z_score=z_score))

	return score


def random_recall_score(random_story_embedding: List[list], recall_embedding: list, correlation=spearmanr, z_score=False) -> float:
	""" Given a list of sentence embeddings for non-corresponding stories, calculate the random recall score with 
	the actuall recall embeddings.

	Args:
		random_story_embedding (list): Sentence embedding vectors for non-correponding stories
		recall_embedding (list): Sentence embedding vectors for recall events
		z_score (bool): z-score matrix (default is False)

	Returns:
		float: Random Recall Score
	"""

	if len(random_story_embedding) > 1:
		random_scores = []
		for i in random_story_embedding:
			score = recall_score(recall_matrix(i, recall_embedding, correlation, z_score=z_score))
			random_scores.append(score)

		random_score = np.mean(random_scores)
	
	else:
		random_score = recall_score(recall_matrix(random_story_embedding, recall_embedding, correlation, z_score=z_score))

	return random_score
	
