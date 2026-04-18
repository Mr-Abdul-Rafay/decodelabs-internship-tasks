"""
Project 3: AI Recommendation Engine - Tech Stack Recommender
Author: DecodeLabs AI Engineer
Description: Content-based filtering using TF-IDF + Cosine Similarity
"""

# ============================================
# STEP 1: DATA MODEL - Job Roles as Items
# ============================================

job_roles = {
    "Cloud Architect": {
        "skills": ["AWS", "Cloud Computing", "Automation", "Python", "Terraform", "Linux"]
    },
    "Data Scientist": {
        "skills": ["Python", "Statistics", "Machine Learning", "SQL", "Pandas", "Data Visualization"]
    },
    "DevOps Engineer": {
        "skills": ["CI/CD", "Docker", "Kubernetes", "Cloud Computing", "Automation", "Linux"]
    },
    "Backend Developer": {
        "skills": ["Python", "Java", "SQL", "REST APIs", "Django", "PostgreSQL"]
    },
    "Frontend Developer": {
        "skills": ["JavaScript", "React", "HTML/CSS", "UI/UX", "Web Design", "TypeScript"]
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "Statistics"]
    },
    "Security Analyst": {
        "skills": ["Cybersecurity", "Network Security", "Linux", "Encryption", "Risk Assessment", "Python"]
    },
    "Mobile Developer": {
        "skills": ["Swift", "Kotlin", "React Native", "iOS", "Android", "Mobile UI"]
    }
}

# ============================================
# STEP 2: COLD START BYPASS - Onboarding Survey
# ============================================

def get_all_unique_skills():
    """Extract all unique skills from all job roles"""
    all_skills = set()
    for role_data in job_roles.values():
        for skill in role_data["skills"]:
            all_skills.add(skill)
    return sorted(list(all_skills))

def onboarding_survey():
    """Collect user preferences to bootstrap the recommendation engine"""
    print("\n" + "="*60)
    print("TECH STACK RECOMMENDER - ONBOARDING SURVEY")
    print("="*60)
    print("\nTo recommend your ideal career path, please tell us about your skills.\n")
    
    all_skills = get_all_unique_skills()
    
    print("Available skills in our database:")
    print("-" * 40)
    for idx, skill in enumerate(all_skills, 1):
        print(f"{idx:2}. {skill}")
    
    print("\n" + "-" * 40)
    print("\nPlease select your top 3 skills:")
    
    user_skills = []
    for i in range(1, 4):
        while True:
            skill_input = input(f"  Skill #{i}: ").strip()
            
            # Case-insensitive matching - FIXED!
            matched_skill = None
            for available_skill in all_skills:
                if available_skill.lower() == skill_input.lower():
                    matched_skill = available_skill
                    break
            
            if matched_skill:
                if matched_skill not in user_skills:
                    user_skills.append(matched_skill)
                    break
                else:
                    print(f"  ⚠️ You already selected '{matched_skill}'. Please choose a different skill.")
            else:
                print(f"  ❌ '{skill_input}' not found in our database.")
                print(f"  Please choose from the list above.")
    
    print("\n" + "="*60)
    print(f"✅ Survey complete! Your skills: {', '.join(user_skills)}")
    print("="*60)
    
    return user_skills

# ============================================
# STEP 2: TF-IDF VECTORIZATION
# ============================================

import math

def calculate_tf(skills_list):
    """
    Term Frequency (TF) = How many times a skill appears in a job role
    For binary presence (simplified), TF = 1 if skill exists, else 0
    But we'll use raw frequency for better weighting
    """
    tf_scores = {}
    for skill in skills_list:
        tf_scores[skill] = skills_list.count(skill)
    return tf_scores

def calculate_idf(all_job_roles):
    """
    Inverse Document Frequency (IDF) = log(Total job roles / Number of roles containing the skill)
    This penalizes common skills (like 'Python') and rewards rare skills
    """
    total_roles = len(all_job_roles)
    skill_document_count = {}
    
    # Count how many job roles contain each skill
    for role_data in all_job_roles.values():
        unique_skills_in_role = set(role_data["skills"])
        for skill in unique_skills_in_role:
            skill_document_count[skill] = skill_document_count.get(skill, 0) + 1
    
    # Calculate IDF
    idf_scores = {}
    for skill, doc_count in skill_document_count.items():
        # Adding 1 to avoid division by zero (not needed here but good practice)
        idf_scores[skill] = math.log(total_roles / doc_count)
    
    return idf_scores

def calculate_tfidf_vector(job_roles_data, idf_scores, user_skills=None):
    """
    Create TF-IDF vectors for each job role (and optionally for user)
    TF-IDF = TF * IDF
    """
    all_skills = get_all_unique_skills()
    vectors = {}
    
    # Create vectors for each job role
    for role_name, role_data in job_roles_data.items():
        vector = []
        for skill in all_skills:
            # Calculate TF (binary: 1 if skill exists, else 0)
            tf = 1 if skill in role_data["skills"] else 0
            # Get IDF (default to 0 if skill not found - shouldn't happen)
            idf = idf_scores.get(skill, 0)
            # TF-IDF weight
            tfidf = tf * idf
            vector.append(tfidf)
        vectors[role_name] = vector
    
    # If user skills provided, create user vector
    user_vector = None
    if user_skills:
        user_vector = []
        for skill in all_skills:
            # For user, TF = 1 if they selected the skill
            tf = 1 if skill in user_skills else 0
            idf = idf_scores.get(skill, 0)
            tfidf = tf * idf
            user_vector.append(tfidf)
    
    return vectors, user_vector, all_skills

def display_vector_sample(vectors, user_vector, all_skills, top_n=10):
    """
    Display first N dimensions of vectors to understand TF-IDF visually
    """
    print("\n" + "="*80)
    print("TF-IDF VECTOR VISUALIZATION (First 10 dimensions)")
    print("="*80)
    
    # Print header
    print(f"\n{'Skill':<20}", end="")
    for role in list(vectors.keys())[:3]:  # Show first 3 roles
        print(f"{role[:15]:<18}", end="")
    print("User Vector")
    print("-"*80)
    
    # Print values for first N skills
    for i in range(min(top_n, len(all_skills))):
        skill = all_skills[i]
        print(f"{skill:<20}", end="")
        
        for role in list(vectors.keys())[:3]:
            value = vectors[role][i]
            print(f"{value:<18.4f}", end="")
        
        if user_vector:
            print(f"{user_vector[i]:<18.4f}")
        else:
            print()
    
    print("\n💡 INTERPRETATION:")
    print("  - Higher values = More important/unique skills")
    print("  - Zero values = Skill not present in that job role")
    print("  - Notice how rare skills get higher weights (e.g., 'Terraform' vs 'Python')")

# ============================================
# STEP 3: COSINE SIMILARITY & RANKING
# ============================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    Formula: (A · B) / (||A|| × ||B||)
    
    Why cosine? It measures ANGLE not MAGNITUDE.
    """
    if not vector_a or not vector_b:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def calculate_all_similarities(role_vectors, user_vector):
    """Compare user vector against all job role vectors"""
    scores = []
    for role_name, role_vector in role_vectors.items():
        score = cosine_similarity(user_vector, role_vector)
        scores.append((role_name, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def get_top_n_recommendations(scores, n=3):
    """Return top N recommendations"""
    return scores[:n]

def display_recommendations(scores, user_skills, n=3):
    """Display formatted recommendations with confidence percentages"""
    print("\n" + "="*60)
    print("🔮 YOUR PERSONALIZED RECOMMENDATIONS")
    print("="*60)
    print(f"\n📌 Based on your skills: {', '.join(user_skills)}")
    print("\n🎯 TOP RECOMMENDED CAREER PATHS:")
    print("-"*60)
    
    top_recommendations = get_top_n_recommendations(scores, n)
    
    for idx, (role_name, score) in enumerate(top_recommendations, 1):
        percentage = score * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"\n{idx}. {role_name}")
        print(f"   Match Confidence: {bar} {percentage:.1f}%")
        print(f"   Similarity Score: {score:.4f}")
    
    print("\n" + "="*60)

def explain_recommendation_logic(scores, role_vectors, user_vector, all_skills):
    """Show WHY a recommendation was made"""
    print("\n" + "="*60)
    print("🧠 WHY THESE RECOMMENDATIONS? (The Math)")
    print("="*60)
    
    top_role = scores[0][0]
    top_score = scores[0][1]
    
    print(f"\nTop match: {top_role} with {top_score*100:.1f}% similarity")
    print("\nMatching skills analysis:")
    
    user_skill_indices = [i for i, val in enumerate(user_vector) if val > 0]
    
    for idx in user_skill_indices:
        skill_name = all_skills[idx]
        role_weight = role_vectors[top_role][idx]
        
        if role_weight > 0:
            print(f"  ✅ {skill_name}: Present in {top_role}")
        else:
            print(f"  ❌ {skill_name}: NOT found in {top_role}")
    
    print("\n📐 Cosine Similarity Formula: similarity = (A·B) / (|A| × |B|)")
    print(f"   = {top_score:.4f}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Step 1: Display database stats
    print("\n📊 DATABASE STATISTICS:")
    print(f"  - Total Job Roles: {len(job_roles)}")
    print(f"  - Total Unique Skills: {len(get_all_unique_skills())}")
    
    # Step 1: Run onboarding survey
    user_profile = onboarding_survey()
    
    print(f"\n📝 User profile created: {user_profile}")
    
    # Step 2: Calculate IDF scores
    print("\n" + "="*60)
    print("STEP 2: CALCULATING TF-IDF VECTORS")
    print("="*60)
    
    idf_scores = calculate_idf(job_roles)
    
    print("\n📊 IDF SCORES (Higher = More unique/important skill):")
    print("-"*40)
    sorted_idf = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n🔝 Most Unique Skills (High IDF):")
    for skill, score in sorted_idf[:5]:
        print(f"  {skill:20} → {score:.4f}")
    print("\n📌 Most Common Skills (Low IDF):")
    for skill, score in sorted_idf[-5:]:
        print(f"  {skill:20} → {score:.4f}")
    
    # Create TF-IDF vectors
    role_vectors, user_vector, all_skills = calculate_tfidf_vector(job_roles, idf_scores, user_profile)
    
    # Display vector preview
    display_vector_sample(role_vectors, user_vector, all_skills, top_n=10)
    
    print("\n" + "="*60)
    print("✅ Step 2 complete! TF-IDF vectors created successfully.")
    print(f"   - {len(role_vectors)} job role vectors created")
    print(f"   - Each vector has {len(all_skills)} dimensions")
    print("="*60)
    
    # ========== STEP 3: COSINE SIMILARITY ==========
    print("\n" + "="*60)
    print("STEP 3: CALCULATING COSINE SIMILARITY")
    print("="*60)
    
    similarity_scores = calculate_all_similarities(role_vectors, user_vector)
    
    print("\n📊 ALL JOB ROLES BY SIMILARITY:")
    print("-"*60)
    for role_name, score in similarity_scores:
        print(f"  {role_name:25} → {score:.4f} ({score*100:.1f}%)")
    
    display_recommendations(similarity_scores, user_profile, n=3)
    explain_recommendation_logic(similarity_scores, role_vectors, user_vector, all_skills)
    
    print("\n" + "="*60)
    print("🎉 PROJECT 3 COMPLETE!")
    print("="*60)
    print("\n✅ Recommendation Engine built with:")
    print("   ✓ Content-Based Filtering")
    print("   ✓ TF-IDF Vectorization")
    print("   ✓ Cosine Similarity")
    print("   ✓ Top-N Ranking")