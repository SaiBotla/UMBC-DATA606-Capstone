## 1. Title and Author

- Developing a Machine Learning-based Recommendation System for TED Talks
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Author Name: SAI KEERTHI BOTLA
- LinkedIn: [http://www.linkedin.com/in/saikeerthibotla)
- GitHub: [https://github.com/SaiBotla)
- Link to your PowerPoint presentation file
    
## 2. Background

TED Talks are popular video presentations covering a wide range of topics, including technology, entertainment, design, science, and business. With thousands of talks available on the TED website, finding the right one to watch can be challenging. A recommendation system can help users find talks that match their interests and preferences. This project proposes the development of a machine learning-based recommendation system for TED Talks.

#### Objective: 
The primary objective of this project is to develop a recommendation system that can recommend TED Talks to users based on their viewing history, preferences, and ratings. The system will use machine learning algorithms to learn from user interactions and make personalized recommendations.


## 3. Data 

The transcripts of all the audio and video recordings of Ted talks posted on Ted.com make up the dataset I'll be utilizing to build a recommendation engine for Ted Talks in this article.
Data Link: https://www.kaggle.com/datasets/rounakbanik/ted-talks?select=ted_main.csv

There are two CSV files.
ted_main.csv - Contains data on actual TED Talk metadata and TED Talk speakers.
transcripts.csv - Contains transcript and URL information for TED Talks

Ted Main has 17 columns and 2550 records:
1.	Comments: The number of first level comments made on the talk
2.	Description: A blurb of what the talk is about
3.	Duration: The duration of the talk in seconds
4.	Event: The TED/TEDx event where the talk took place
5.	film_date: The Unix timestamp of the filming
6.	languages: The number of languages in which the talk is available.
7.	main_speaker: The first named speaker of the talk
8.	name: The official name of the TED Talk. Includes the title and the speaker.
9.	num_speaker: The number of speakers in the talk
10.	published_date: The Unix timestamp for the publication of the talk on TED.com.
11.	ratings: A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.)
12.	related_talks: A list of dictionaries of recommended talks to watch next.
13.	speaker_occupation: The occupation of the main speaker.
14.	tags: The themes associated with the talk.
15.	title: The title of the talk.
16.	url: The URL of the talk.
17.	views: The number of views on the talk.

##### Transcripts has Two columns, and both the columns are Text:
1.Transcript (Transcript of the Ted Talk) 
2.URL (link to the Ted Talk)
Data has 2464 Records.

## Expected Outcome: 
The proposed recommendation system will enable users to discover TED Talks that match their interests and preferences. The system will provide accurate and personalized recommendations using machine learning algorithms. The project will contribute to the development of intelligent systems that can learn from user interactions and make personalized recommendations.
