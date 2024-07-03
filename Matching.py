import spacy
import fitz
import io
from flask import session, request
from database import mongo
from bson.objectid import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumeFetchedData = mongo.db.resumeFetchedData
JOBS = mongo.db.JOBS

### Spacy model
print("Loading Jd Parser model...")
jd_model = spacy.load('assets/JdModel/output/model-best')
print("Jd Parser model loaded")

def preprocess_text(text):
    # Preprocess the text using spaCy for tokenization and lemmatization
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    processed_text = " ".join([token.lemma_ for token in doc])
    return processed_text   

def calculate_cosine_similarity(doc1, doc2):
    # Calculate cosine similarity between two documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)[0, 1]
    return cosine_sim
    
def Matching():
    job_id = request.form['job_id']
    jd_data = JOBS.find_one({"_id": ObjectId(job_id)}, {"FileData": 1})["FileData"]

    with io.BytesIO(jd_data) as data:
        doc = fitz.open(stream=data)
        text_of_jd = " "
        for page in doc:
            text_of_jd = text_of_jd + str(page.get_text())

    label_list_jd = []
    text_list_jd = []
    dic_jd = {}

    doc_jd = jd_model(text_of_jd)
    for ent in doc_jd.ents:
        label_list_jd.append(ent.label_)
        text_list_jd.append(ent.text)

    print("Model work done")

    for i in range(len(label_list_jd)):
        if label_list_jd[i] in dic_jd:
            dic_jd[label_list_jd[i]].append(text_list_jd[i])
        else:
            dic_jd[label_list_jd[i]] = [text_list_jd[i]]

    print("Jd dictionary:", dic_jd)

    resume_workedAs = resumeFetchedData.find_one({"UserId": ObjectId(session['user_id'])}, {"WORKED AS": 1})["WORKED AS"]
    print("resume_workedAs: ", resume_workedAs)



    resume_experience_list = resumeFetchedData.find_one({"UserId": ObjectId(session['user_id'])}, {"YEARS OF EXPERIENCE": 1})["YEARS OF EXPERIENCE"]
    print("resume_experience: ", resume_experience_list)

    resume_skills = resumeFetchedData.find_one({"UserId": ObjectId(session['user_id'])}, {"SKILLS": 1})["SKILLS"]
    print("resume_skills: ", resume_skills)

    job_description_skills = dic_jd.get('SKILLS')
    print("job_description_skills: ", job_description_skills)

    # Check if job_description_skills is not None before attempting to iterate
    if job_description_skills is not None:
        jd_experience_list = dic_jd.get('EXPERIENCE')
        print("jd_experience_list: ", jd_experience_list)

        # Check if jd_experience_list is not None before attempting to iterate
        jd_experience = []
        if jd_experience_list is not None:

            for p in jd_experience_list:
                parts = p.split()
                if "years" in p or "year" in p:
                    year = int(parts[0])
                    if "months" in p or "month" in p:
                        year += int(parts[2]) / 12
                else:
                    year = int(parts[0]) / 12
                year = round(year, 2)
                jd_experience.append(year)

            print("jd_experience: ", jd_experience)
        else:
            print("No job experience found in the job description.")
            # You might want to set a default value or handle this case appropriately.
            skip_similarity_calculation = True

        jd_post = dic_jd.get('JOBPOST')
        print("jd_post: ", jd_post)

        ###########################################################
        #########  Compare resume_workedAs and jd_post
        jd_post = [item.lower() for item in jd_post]
        experience_similarity = 0
        match_index = -1
        jdpost_similarity = 0

        if resume_workedAs:
            resume_workedAs = [item.lower() for item in resume_workedAs]

            for i, item in enumerate(resume_workedAs):
                if item in jd_post:
                    result = True
                    match_index = i
                    ########   compare resume_experience and jd_experience
                    max_experience_similarity = 0  # Initialize maximum experience similarity
                    resume_experience = []
                    if resume_experience_list:
                        # resume_experience = []

                        for p in resume_experience_list:
                            parts = p.split()
                            if "years" in p or "year" in p:
                                year = int(parts[0])
                                if "months" in p or "month" in p:
                                    year += int(parts[2]) / 12
                            else:
                                year = int(parts[0]) / 12
                            year = round(year, 2)
                            resume_experience.append(year)

                        print("resume_experience: ", resume_experience)
                    else:
                        # print("No years of experience found in the resume.")
                        print("")
                        # You might want to set a default value or handle this case appropriately.

                    # Adjusted experience similarity calculation
                    if resume_experience and jd_experience:
                        for resume_exp in resume_experience:
                            if resume_exp >= jd_experience[0]:  # Compare each element with jd_experience
                                print("Experience Matched")
                                experience_similarity = 1
                            elif 0 < (resume_exp - jd_experience[0]) <= 1:
                                print("Experience can be considered")
                                experience_similarity = 0.7
                            else:
                                print("Experience Unmatched")
                                experience_similarity = 0

                            if experience_similarity > max_experience_similarity:
                                max_experience_similarity = experience_similarity

                        print("Highest Experience Similarity:", max_experience_similarity)

                    else:
                        print("No years of experience found in the resume.")

                    break
                else:
                    result = False

            if result:
                jdpost_similarity = 1
            else:
                jdpost_similarity = 0
                
        jdpost_similarity = jdpost_similarity * 0.3
        print("jd_post_simiarity: ", jdpost_similarity)
        experience_similarity = max_experience_similarity * 0.2 if 'max_experience_similarity' in locals() else 0
        print("Experiece Similarity: ", experience_similarity)
        ########   compare resume_skills and jd_skills
        if resume_skills and job_description_skills:
            processed_resume_skills = preprocess_text(", ".join(resume_skills))
            processed_jd_skills = preprocess_text(", ".join(job_description_skills))

            skills_similarity = calculate_cosine_similarity(processed_resume_skills, processed_jd_skills)
            # print("Skills Similarity original out of function: ", skills_similarity)

            # if skills_similarity > 0.3:
            #     skills_similarity = (skills_similarity * 1.4) * 0.5
            # else:
            #     skills_similarity = skills_similarity * 0.5
            # print("Skills Similarity: ", skills_similarity)
            skills_similarity = skills_similarity * 0.5
            print("Skills Similarity: ", skills_similarity)
        else:
            skills_similarity = 0
            print("Skills Similarity: ", skills_similarity)

        if 'skip_similarity_calculation' in locals():
            # Calculate similarity without considering jd_post and experience_similarity
            skills_similarity = skills_similarity / 0.5;
            # print("Skills Similarity after /0.5: ", skills_similarity)
            skills_similarity = skills_similarity * 0.7;
            print("Skills Similarity recalculated: ", skills_similarity)
            matching = (jdpost_similarity + skills_similarity) * 100
        else:
            matching = (jdpost_similarity + experience_similarity + skills_similarity) * 100

        matching = round(matching, 2)
        print("Overall Similarity between resume and jd is ", matching)

        return matching

    else:
        print("No skills found in the job description.")
        # You might want to set a default value or handle this case appropriately.

# Example usage:
# result = Matching()
# print("Matching result:", result)
