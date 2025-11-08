import json
import os
import time
import hashlib
from typing import List, Dict, Optional
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# LinkedIn scraping imports
try:
    from linkedin_scraper import Person, actions
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import WebDriverException
    LINKEDIN_AVAILABLE = True
except ImportError:
    LINKEDIN_AVAILABLE = False
    print("⚠️ LinkedIn scraping dependencies not available. Scraping feature will be disabled.")

app = Flask(__name__)
CORS(app)

def create_html_template():
    """Create the HTML template file"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Profiles RAG System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .sidebar {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-panel {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
            margin-bottom: 15px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-box {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            min-height: 60px;
        }
        
        .example-questions {
            background: #f7fafc;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .example-questions h4 {
            margin-bottom: 15px;
            color: #4a5568;
        }
        
        .example-questions ul {
            list-style: none;
            padding: 0;
        }
        
        .example-questions li {
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
            color: #666;
        }
        
        .example-questions li:last-child {
            border-bottom: none;
        }
        
        .question-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            margin-bottom: 20px;
            resize: vertical;
            min-height: 100px;
        }
        
        .question-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .linkedin-input-section {
            margin-bottom: 20px;
        }
        
        .linkedin-urls-input {
            width: 100%;
            min-height: 100px;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 0.9rem;
            font-family: inherit;
            resize: vertical;
            margin-bottom: 10px;
            background-color: #f8f9fa;
        }
        
        .linkedin-urls-input:focus {
            outline: none;
            border-color: #667eea;
            background-color: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .answer-box {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
                 .success { color: #38a169; }
         .error { color: #e53e3e; }
         .info { color: #3182ce; }
         
         .notification {
             position: fixed;
             top: 20px;
             right: 20px;
             background: #38a169;
             color: white;
             padding: 15px 20px;
             border-radius: 8px;
             box-shadow: 0 4px 12px rgba(0,0,0,0.15);
             z-index: 1000;
             transform: translateX(400px);
             transition: transform 0.3s ease;
         }
         
         .notification.show {
             transform: translateX(0);
         }
         
         .notification.info {
             background: #3182ce;
         }
         
         .notification.error {
             background: #e53e3e;
         }
        
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        #skillsChart {
            max-height: 300px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LinkedIn Profiles RAG System</h1>
            <p>Ask questions about LinkedIn profiles and get intelligent answers based on the data</p>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                                 <h3 class="section-title">🚀 System Setup</h3>
                 <button class="btn" id="setupBtn">Setup RAG System</button>
                 <div class="status-box" id="setupStatus">
                     Click "Setup RAG System" to initialize the system
                 </div>
                                   <div class="info-box" style="background: #e6f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 15px; margin-bottom: 20px; font-size: 0.9rem;">
                      <strong>💡 Tip:</strong> If you encounter GPU memory errors, try installing a smaller model:<br>
                      <code>ollama pull llama2:3b</code> or <code>ollama pull llama2:7b</code>
                  </div>
                  <div class="info-box" style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin-bottom: 20px; font-size: 0.9rem;">
                      <strong>🔧 ChromeDriver:</strong> For LinkedIn scraping, ensure Chrome is installed and run:<br>
                      <code>pip install webdriver-manager</code>
                  </div>
                
                <h3 class="section-title">📊 Database Info</h3>
                <button class="btn" id="summaryBtn">Get Database Summary</button>
                <div class="status-box" id="summaryStatus">
                    Click to get database summary
                </div>
                
                <div class="chart-container" id="chartContainer" style="display: none;">
                    <canvas id="skillsChart" width="400" height="200"></canvas>
                </div>
                
                <h3 class="section-title">🔗 Add LinkedIn Profiles</h3>
                <div class="linkedin-input-section">
                    <textarea 
                        class="linkedin-urls-input" 
                        id="linkedinUrlsInput" 
                        placeholder="Enter LinkedIn profile URLs (one per line)&#10;Example:&#10;https://linkedin.com/in/username1&#10;https://linkedin.com/in/username2"
                    ></textarea>
                    <button class="btn" id="scrapeBtn">Scrape Profiles</button>
                    <div class="status-box" id="scrapeStatus">
                        Enter LinkedIn URLs and click "Scrape Profiles"
                    </div>
                </div>
                

            </div>
            
            <div class="main-panel">
                <h3 class="section-title">💬 Ask Questions</h3>
                
                <div class="example-questions">
                    <h4>Example questions:</h4>
                    <ul>
                        <li>Who has Python skills?</li>
                        <li>Who knows machine learning?</li>
                        <li>Find people with AI experience</li>
                        <li>Who works at NxtGen Cloud Technologies?</li>
                        <li>Who has data science skills?</li>
                        <li>Who knows React or Angular?</li>
                        <li>Find people with cloud computing experience</li>
                    </ul>
                </div>
                
                <textarea 
                    class="question-input" 
                    id="questionInput" 
                    placeholder="Ask anything about the LinkedIn profiles..."
                ></textarea>
                
                <button class="btn" id="askBtn" disabled>Ask Question</button>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    Processing your question...
                </div>
                
                <div class="answer-box" id="answerBox">
                    Your answer will appear here...
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Global state
        let systemReady = false;
        
        // DOM elements
        const setupBtn = document.getElementById('setupBtn');
        const setupStatus = document.getElementById('setupStatus');
        const summaryBtn = document.getElementById('summaryBtn');
        const summaryStatus = document.getElementById('summaryStatus');
        const chartContainer = document.getElementById('chartContainer');
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');
        const loading = document.getElementById('loading');
        const answerBox = document.getElementById('answerBox');
        const linkedinUrlsInput = document.getElementById('linkedinUrlsInput');
        const scrapeBtn = document.getElementById('scrapeBtn');
        const scrapeStatus = document.getElementById('scrapeStatus');
        
        // Setup system
        setupBtn.addEventListener('click', async () => {
            setupBtn.disabled = true;
            setupBtn.textContent = 'Setting up...';
            setupStatus.textContent = 'Initializing system...';
            
            try {
                const response = await fetch('/api/setup', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    setupStatus.innerHTML = '<span class="success">✅ ' + data.message + '</span>';
                    systemReady = true;
                    askBtn.disabled = false;
                    askBtn.textContent = 'Ask Question';
                } else {
                    setupStatus.innerHTML = '<span class="error">❌ ' + data.message + '</span>';
                }
            } catch (error) {
                setupStatus.innerHTML = '<span class="error">❌ Setup failed: ' + error.message + '</span>';
            } finally {
                setupBtn.disabled = false;
                setupBtn.textContent = 'Setup RAG System';
            }
        });
        
        // Get summary
        summaryBtn.addEventListener('click', async () => {
            summaryBtn.disabled = true;
            summaryBtn.textContent = 'Loading...';
            summaryStatus.textContent = 'Fetching summary...';
            
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();
                
                if (data.success) {
                    const summary = data.summary;
                    let summaryText = `📊 Total Profiles: ${summary.total_profiles}\\n\\n`;
                    
                    if (summary.top_skills && Object.keys(summary.top_skills).length > 0) {
                        summaryText += '🔧 Top Skills Found:\\n';
                        Object.entries(summary.top_skills).forEach(([skill, count]) => {
                            summaryText += `  ${skill}: ${count} profiles\\n`;
                        });
                        
                        // Check if Chart.js is available before creating chart
                        if (typeof Chart !== 'undefined') {
                            // Show chart container and create bar chart
                            chartContainer.style.display = 'block';
                            try {
                                createSkillsChart(summary.top_skills);
                            } catch (chartError) {
                                console.warn('Chart creation failed, using text display:', chartError);
                                createSkillsTextDisplay(summary.top_skills);
                            }
                        } else {
                            console.warn('Chart.js not loaded, using text display');
                            chartContainer.style.display = 'block';
                            createSkillsTextDisplay(summary.top_skills);
                        }
                    } else {
                        summaryText += 'No skills data available';
                        chartContainer.style.display = 'none';
                    }
                    
                    summaryStatus.textContent = summaryText;
                } else {
                    summaryStatus.innerHTML = '<span class="error">❌ ' + data.message + '</span>';
                    chartContainer.style.display = 'none';
                }
            } catch (error) {
                summaryStatus.innerHTML = '<span class="error">❌ Failed to get summary: ' + error.message + '</span>';
                chartContainer.style.display = 'none';
            } finally {
                summaryBtn.disabled = false;
                summaryBtn.textContent = 'Get Database Summary';
            }
        });
        
        // Scrape LinkedIn profiles
        scrapeBtn.addEventListener('click', async () => {
            const urls = linkedinUrlsInput.value.trim();
            if (!urls) {
                scrapeStatus.innerHTML = '<span class="error">❌ Please enter LinkedIn profile URLs</span>';
                return;
            }
            
            const urlList = urls.split('\\n').filter(url => url.trim());
            if (urlList.length === 0) {
                scrapeStatus.innerHTML = '<span class="error">❌ No valid URLs found</span>';
                return;
            }
            
                         scrapeBtn.disabled = true;
             scrapeBtn.textContent = 'Scraping...';
             scrapeStatus.innerHTML = '<span class="info">🔄 Starting to scrape ' + urlList.length + ' profile(s)...</span>';
             showNotification(`Starting to scrape ${urlList.length} profile(s)...`, 'info');
            
            try {
                const response = await fetch('/api/scrape', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        urls: urlList
                    })
                });
                
                const data = await response.json();
                
                                 if (data.success) {
                     scrapeStatus.innerHTML = '<span class="success">✅ ' + data.message + '</span>';
                     // Clear the input after successful scraping
                     linkedinUrlsInput.value = '';
                     // Refresh the system if it was already set up
                     if (systemReady) {
                         setupStatus.innerHTML = '<span class="info">ℹ️ New data added. Consider re-setting up the system for updated results.</span>';
                     }
                     
                     // Automatically update the database summary
                     scrapeStatus.innerHTML = '<span class="success">✅ ' + data.message + '</span><br><span class="info">🔄 Updating database summary...</span>';
                     await updateDatabaseSummary();
                     scrapeStatus.innerHTML = '<span class="success">✅ ' + data.message + '</span><br><span class="success">✅ Database summary updated!</span>';
                 } else {
                     scrapeStatus.innerHTML = '<span class="error">❌ ' + data.message + '</span>';
                 }
            } catch (error) {
                scrapeStatus.innerHTML = '<span class="error">❌ Scraping failed: ' + error.message + '</span>';
            } finally {
                scrapeBtn.disabled = false;
                scrapeBtn.textContent = 'Scrape Profiles';
            }
                 });
         
         // Function to show notifications
         function showNotification(message, type = 'success') {
             // Remove existing notifications
             const existingNotifications = document.querySelectorAll('.notification');
             existingNotifications.forEach(notification => notification.remove());
             
             // Create new notification
             const notification = document.createElement('div');
             notification.className = `notification ${type}`;
             notification.textContent = message;
             
             // Add to page
             document.body.appendChild(notification);
             
             // Show notification
             setTimeout(() => {
                 notification.classList.add('show');
             }, 100);
             
             // Hide notification after 3 seconds
             setTimeout(() => {
                 notification.classList.remove('show');
                 setTimeout(() => {
                     if (notification.parentNode) {
                         notification.parentNode.removeChild(notification);
                     }
                 }, 300);
             }, 3000);
         }
         
         // Function to automatically update database summary
         async function updateDatabaseSummary() {
             try {
                 summaryStatus.textContent = '🔄 Updating summary...';
                 
                 const response = await fetch('/api/summary');
                 const data = await response.json();
                 
                 if (data.success) {
                     const summary = data.summary;
                     let summaryText = `📊 Total Profiles: ${summary.total_profiles}\\n\\n`;
                     
                     if (summary.top_skills && Object.keys(summary.top_skills).length > 0) {
                         summaryText += '🔧 Top Skills Found:\\n';
                         Object.entries(summary.top_skills).forEach(([skill, count]) => {
                             summaryText += `  ${skill}: ${count} profiles\\n`;
                         });
                         
                         // Update chart with new data
                         if (typeof Chart !== 'undefined') {
                             chartContainer.style.display = 'block';
                             try {
                                 createSkillsChart(summary.top_skills);
                             } catch (chartError) {
                                 console.warn('Chart update failed, using text display:', chartError);
                                 createSkillsTextDisplay(summary.top_skills);
                             }
                         } else {
                             chartContainer.style.display = 'block';
                             createSkillsTextDisplay(summary.top_skills);
                         }
                     } else {
                         summaryText += 'No skills data available';
                         chartContainer.style.display = 'none';
                     }
                     
                     summaryStatus.textContent = summaryText;
                     console.log('✅ Database summary updated automatically');
                     showNotification('Database summary updated successfully!', 'success');
                 } else {
                     console.warn('⚠️ Failed to update summary automatically:', data.message);
                     showNotification('Failed to update summary', 'error');
                 }
             } catch (error) {
                 console.error('❌ Error updating summary automatically:', error);
             }
         }
         
         // Function to create skills bar chart
        function createSkillsChart(skillsData) {
            try {
                const canvas = document.getElementById('skillsChart');
                if (!canvas) {
                    console.error('Canvas element not found');
                    return;
                }
                
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    console.error('Could not get 2D context');
                    return;
                }
                
                // Safely destroy existing chart if it exists
                if (window.skillsChart && typeof window.skillsChart.destroy === 'function') {
                    try {
                        window.skillsChart.destroy();
                    } catch (e) {
                        console.log('Error destroying existing chart:', e);
                    }
                }
                
                const labels = Object.keys(skillsData);
                const data = Object.values(skillsData);
                
                // Create new chart
                window.skillsChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Number of Profiles',
                            data: data,
                            backgroundColor: 'rgba(102, 126, 234, 0.8)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    stepSize: 1
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
                
                console.log('Chart created successfully');
            } catch (error) {
                console.error('Error creating chart:', error);
                // Hide chart container if chart creation fails
                const chartContainer = document.getElementById('chartContainer');
                if (chartContainer) {
                    chartContainer.style.display = 'none';
                }
            }
        }
        
        // Alternative function to create a simple text-based skills display
        function createSkillsTextDisplay(skillsData) {
            const chartContainer = document.getElementById('chartContainer');
            if (!chartContainer) return;
            
            // Clear existing content
            chartContainer.innerHTML = '';
            
            // Create a simple text display
            const textDisplay = document.createElement('div');
            textDisplay.style.padding = '20px';
            textDisplay.style.textAlign = 'center';
            
            let textContent = '<h4>📊 Skills Distribution</h4><div style="margin-top: 15px;">';
            
            Object.entries(skillsData).forEach(([skill, count]) => {
                const barWidth = (count / Math.max(...Object.values(skillsData))) * 200;
                textContent += `
                    <div style="margin: 10px 0; text-align: left;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span><strong>${skill}</strong></span>
                            <span>${count} profiles</span>
                        </div>
                        <div style="background: #e2e8f0; height: 20px; border-radius: 10px; overflow: hidden;">
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: ${barWidth}px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                `;
            });
            
            textContent += '</div>';
            textDisplay.innerHTML = textContent;
            chartContainer.appendChild(textDisplay);
        }
        


        // Ask question
        askBtn.addEventListener('click', async () => {
            const question = questionInput.value.trim();
            if (!question) {
                alert('Please enter a question.');
                return;
            }
            
            if (!systemReady) {
                alert('Please setup the RAG system first.');
                return;
            }
            
            askBtn.disabled = true;
            askBtn.textContent = 'Processing...';
            loading.style.display = 'block';
            answerBox.textContent = 'Processing your question...';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    answerBox.textContent = data.answer;
                } else {
                    answerBox.innerHTML = '<span class="error">❌ ' + data.message + '</span>';
                }
            } catch (error) {
                answerBox.innerHTML = '<span class="error">❌ Error: ' + error.message + '</span>';
            } finally {
                askBtn.disabled = false;
                askBtn.textContent = 'Ask Question';
                loading.style.display = 'none';
            }
        });
        
        // Enter key support
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                askBtn.click();
            }
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            console.log('LinkedIn RAG System loaded successfully!');
        });
    </script>
</body>
</html>"""
    
    # Write the HTML template
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML template created successfully!")

class LinkedInRAGApp:
    def __init__(self, json_file_path: str):
        """Initialize the RAG application with LinkedIn profiles data"""
        self.json_file_path = json_file_path
        self.profiles_data = self._load_profiles()
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        
        # LinkedIn scraping configuration
        self.linkedin_email = "gaurav30.kumavat@gmail.com"
        self.linkedin_password = "@G@ur@v10"
        
    def _load_profiles(self) -> List[Dict]:
        """Load the LinkedIn profiles from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"✅ Loaded {len(data)} LinkedIn profiles")
                return data
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return []
    
    def _prepare_documents(self) -> List[Document]:
        """Convert LinkedIn profiles into LangChain documents"""
        documents = []
        
        for profile in self.profiles_data:
            # Create a comprehensive text representation of each profile
            profile_text = self._profile_to_text(profile)
            
            # Create metadata for the document
            metadata = {
                "name": profile.get("name", "Unknown"),
                "linkedin_url": profile.get("linkedin_url", ""),
                "profile_type": "linkedin_profile"
            }
            
            # Create LangChain Document
            doc = Document(page_content=profile_text, metadata=metadata)
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} documents from profiles")
        return documents
    
    def _profile_to_text(self, profile: Dict) -> str:
        """Convert a LinkedIn profile to searchable text"""
        text_parts = []
        
        # Basic info
        name = profile.get("name", "")
        if name:
            text_parts.append(f"Name: {name}")
        
        # About section
        about = profile.get("about", "")
        if about:
            text_parts.append(f"About: {about}")
        
        # Experience section
        experiences = profile.get("experiences", [])
        if experiences:
            text_parts.append("Experience:")
            for exp in experiences:
                exp_text = []
                if exp.get("position_title"):
                    exp_text.append(f"Position: {exp['position_title']}")
                if exp.get("institution_name"):
                    exp_text.append(f"Company: {exp['institution_name']}")
                if exp.get("description"):
                    exp_text.append(f"Description: {exp['description']}")
                if exp.get("duration"):
                    exp_text.append(f"Duration: {exp['duration']}")
                if exp.get("location"):
                    exp_text.append(f"Location: {exp['location']}")
                
                if exp_text:
                    text_parts.append("  - " + " | ".join(exp_text))
        
        # Education section
        education = profile.get("education", [])
        if education:
            text_parts.append("Education:")
            for edu in education:
                edu_text = []
                if edu.get("degree"):
                    edu_text.append(f"Degree: {edu['degree']}")
                if edu.get("institution_name"):
                    edu_text.append(f"Institution: {edu['institution_name']}")
                if edu.get("description"):
                    edu_text.append(f"Description: {edu['description']}")
                if edu.get("from_date") and edu.get("to_date"):
                    edu_text.append(f"Period: {edu['from_date']} to {edu['to_date']}")
                
                if edu_text:
                    text_parts.append("  - " + " | ".join(edu_text))
        
        return "\n".join(text_parts)
    
    def setup_vectorstore(self):
        """Set up the vector store with embeddings"""
        try:
            print("🔧 Setting up vector store...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Prepare documents
            documents = self._prepare_documents()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"✅ Split into {len(chunks)} chunks")
            
            # Clear existing ChromaDB directory to avoid compatibility issues
            import shutil
            import os
            if os.path.exists("./chroma_db"):
                print("🗑️ Removing existing ChromaDB directory...")
                shutil.rmtree("./chroma_db")
            
            # Create vector store with specific ChromaDB settings
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db",
                client_settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            print("✅ Vector store created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up vector store: {e}")
            print("🔄 Trying alternative approach...")
            
            try:
                # Alternative approach: Create a new ChromaDB instance
                import shutil
                import os
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                # Create vector store without persist_directory first
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                
                print("✅ Vector store created successfully (alternative method)!")
                return True
                
            except Exception as e2:
                print(f"❌ Alternative approach also failed: {e2}")
                return False
    
    def setup_qa_chain(self):
        """Set up the question-answering chain"""
        try:
            print("🔧 Setting up QA chain...")
            
            # Check if vector store is properly initialized
            if not self.vectorstore:
                print("❌ Vector store not initialized. Cannot create QA chain.")
                return False

                 # Initialize Ollama LLM
            llm = Ollama(model="llama3.2:1b")
            
            # Initialize Ollama LLM with CPU-only model to avoid GPU memory issues
            # try:
            #     # Try to use a smaller model first
            #     llm = Ollama(model="llama3.2:1b", temperature=0.1)
            #     print("✅ Using llama3.2:1b model")
            # except Exception as e:
            #     print(f"⚠️ Failed to load llama3.2:1b, trying CPU-only model: {e}")
            #     try:
            #         # Use a CPU-only model
            #         llm = Ollama(model="llama3.2:1b", temperature=0.1, device="cpu")
            #         print("✅ Using llama3.2:1b with CPU device")
            #     except Exception as e2:
            #         print(f"⚠️ Failed to load llama3.2:1b with CPU, trying tiny model: {e2}")
            #         try:
            #             # Use the smallest available model
            #             llm = Ollama(model="llama3.2:1b", temperature=0.1)
            #             print("✅ Using llama3.2:1b model")
            #         except Exception as e3:
            #             print(f"⚠️ Failed to load llama3.2:1b, trying CPU-only tiny model: {e3}")
            #             try:
            #                 # Use the smallest model with CPU only
            #                 llm = Ollama(model="llama3.2:1b", temperature=0.1, device="cpu")
            #                 print("✅ Using llama3.2:1b with CPU device")
            #             except Exception as e4:
            #                 print(f"❌ All Ollama models failed. Using fallback approach: {e4}")
            #                 # Create a simple fallback that doesn't require Ollama
            #                 return self._setup_fallback_qa_chain()
            
            # Create custom prompt template
            prompt_template = """You are a helpful assistant that answers questions about LinkedIn profiles and professional skills. 
            Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

            IMPORTANT: When asked about skills, technologies, or specific qualifications:
            1. Always list the NAMES of people who have them
            2. Provide specific evidence from their experience, about section, or education
            3. Be specific about where/how they acquired these skills
            4. If multiple people have the skill, mention ALL of them

            Format your response to clearly show who has what skills with evidence. For example:
            - "Neha M has Python skills based on her work as a Developer at Tech Company"
            - "Amee Popat mentions AI/ML in her about section and studied Computer Science"
            - "John Doe has machine learning experience from his role as Data Scientist at AI Corp"

            Always mention the person's name when discussing their skills or experience.

            Context: {context}

            Question: {question}

            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt}
            )
            
            print("✅ QA chain created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up QA chain: {e}")
            return False
    
    def _setup_fallback_qa_chain(self):
        """Setup a fallback QA chain that doesn't require Ollama"""
        try:
            print("🔧 Setting up fallback QA chain (no LLM required)...")
            
            # Create a simple fallback that uses direct text search
            self.qa_chain = self._create_fallback_chain()
            
            print("✅ Fallback QA chain created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up fallback QA chain: {e}")
            return False
    
    def _create_fallback_chain(self):
        """Create a simple fallback chain that doesn't use LLM"""
        class FallbackChain:
            def __init__(self, rag_app):
                self.rag_app = rag_app
            
            def run(self, question):
                # Use the existing skill analysis method
                skill_analysis = self.rag_app._analyze_skills_by_section(question)
                
                if skill_analysis:
                    result = f"📋 **Skills Analysis (Fallback Mode):**\n\n"
                    
                    for skill, people_data in skill_analysis.items():
                        result += f"🔧 **{skill.upper()}**\n"
                        
                        if people_data['experience']:
                            experience_names = list(people_data['experience'].keys())
                            result += f"  📁 **Based on Experience**: {', '.join(experience_names)}\n"
                        
                        if people_data['about']:
                            about_names = list(people_data['about'].keys())
                            result += f"  📝 **Based on About Section**: {', '.join(about_names)}\n"
                        
                        if people_data['education']:
                            education_names = list(people_data['education'].keys())
                            result += f"  🎓 **Based on Education**: {', '.join(education_names)}\n"
                        
                        result += "\n"
                    
                    # Add summary of all people found
                    all_people = set()
                    for skill_data in skill_analysis.values():
                        all_people.update(skill_data['experience'].keys())
                        all_people.update(skill_data['about'].keys())
                        all_people.update(skill_data['education'].keys())
                    
                    if all_people:
                        result += f"👥 **Total People Found: {len(all_people)}**\n"
                        result += f"**Names**: {', '.join(sorted(all_people))}\n\n"
                    
                    result += f"🤖 Note: This is a fallback response (no LLM available). For more detailed analysis, please ensure Ollama is running with a compatible model."
                    return result
                else:
                    return "❌ No relevant information found in the profiles for this question. (Fallback mode - no LLM available)"
        
        return FallbackChain(self)
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question"""
        try:
            if not self.qa_chain:
                return "❌ QA chain not initialized. Please set up the system first."
            
            if not self.vectorstore:
                return "❌ Vector store not initialized. Please set up the system first."
            
            # Get answer from the chain
            result = self.qa_chain.run(question)
            
            # Enhance the response to highlight names for skill-based questions
            if any(keyword in question.lower() for keyword in ['who has', 'who knows', 'who can', 'find people', 'people with', 'who works with']):
                result = self._enhance_response_with_names(result, question)
            
            return result
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def _enhance_response_with_names(self, response: str, question: str) -> str:
        """Enhance response to better highlight names and skills with evidence"""
        # Get detailed skill analysis for the question
        skill_analysis = self._analyze_skills_by_section(question)
        
        if skill_analysis:
            # Build simplified response with just names
            result = f"📋 Skills Analysis:\n\n"
            
            for skill, people_data in skill_analysis.items():
                result += f"🔧 {skill.upper()}\n"
                
                if people_data['experience']:
                    experience_names = list(people_data['experience'].keys())
                    result += f"  📁 Based on Experience: {', '.join(experience_names)}\n"
                
                if people_data['about']:
                    about_names = list(people_data['about'].keys())
                    result += f"  📝 Based on About Section: {', '.join(about_names)}\n"
                
                if people_data['education']:
                    education_names = list(people_data['education'].keys())
                    result += f"  🎓 Based on Education: {', '.join(education_names)}\n"
                
                result += "\n"
            
            # Add summary of all people found
            all_people = set()
            for skill_data in skill_analysis.values():
                all_people.update(skill_data['experience'].keys())
                all_people.update(skill_data['about'].keys())
                all_people.update(skill_data['education'].keys())
            
            if all_people:
                result += f"👥 Total People Found: {len(all_people)}\n"
                result += f"Names: {', '.join(sorted(all_people))}\n\n"
            
            result += f"{response}"
            return result
        
        return response
    
    def _analyze_skills_by_section(self, question: str) -> Dict:
        """Analyze skills by different sections of profiles"""
        # Extract skills from the question
        question_skills = self._extract_skills_from_question(question)
        
        if not question_skills:
            return {}
        
        skill_analysis = {}
        
        for skill in question_skills:
            skill_analysis[skill] = {
                'experience': {},
                'about': {},
                'education': {}
            }
            
            for profile in self.profiles_data:
                name = profile.get("name", "Unknown")
                
                # Check experience section
                experiences = profile.get("experiences", [])
                for exp in experiences:
                    exp_text = f"{exp.get('position_title', '')} {exp.get('institution_name', '')} {exp.get('description', '')}".lower()
                    if skill.lower() in exp_text:
                        evidence = f"Works as {exp.get('position_title', 'Unknown role')} at {exp.get('institution_name', 'Unknown company')}"
                        if exp.get('duration'):
                            evidence += f" ({exp.get('duration')})"
                        skill_analysis[skill]['experience'][name] = evidence
                
                # Check about section
                about = profile.get("about", "")
                if about and skill.lower() in about.lower():
                    # Extract relevant sentence containing the skill
                    sentences = about.split('.')
                    relevant_sentence = ""
                    for sentence in sentences:
                        if skill.lower() in sentence.lower():
                            relevant_sentence = sentence.strip()
                            break
                    if relevant_sentence:
                        skill_analysis[skill]['about'][name] = f"\"{relevant_sentence[:100]}{'...' if len(relevant_sentence) > 100 else ''}\""
                
                # Check education section
                education = profile.get("education", [])
                for edu in education:
                    edu_text = f"{edu.get('degree', '')} {edu.get('institution_name', '')} {edu.get('description', '')}".lower()
                    if skill.lower() in edu_text:
                        evidence = f"Studied {edu.get('degree', 'Unknown degree')} at {edu.get('institution_name', 'Unknown institution')}"
                        if edu.get('description'):
                            evidence += f" - {edu.get('description', '')}"
                        skill_analysis[skill]['education'][name] = evidence
        
        # Remove skills with no evidence
        skill_analysis = {skill: data for skill, data in skill_analysis.items() 
                         if any(data.values())}
        
        return skill_analysis
    
    def _extract_skills_from_question(self, question: str) -> List[str]:
        """Extract skills mentioned in the question"""
        question_lower = question.lower()
        
        # Common skills to look for
        skills = [
            'python', 'java', 'javascript', 'ai', 'ml', 'machine learning',
            'deep learning', 'data science', 'web development', 'cloud',
            'sql', 'react', 'angular', 'node.js', 'django', 'flask',
            'artificial intelligence', 'neural networks', 'computer vision',
            'natural language processing', 'big data', 'hadoop', 'spark',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'devops',
            'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'scala',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'plotly', 'jupyter', 'git', 'github'
        ]
        
        found_skills = []
        for skill in skills:
            if skill in question_lower:
                found_skills.append(skill)
        
        return found_skills
    

    
    def get_profile_summary(self) -> Dict:
        """Get a summary of all profiles in the database"""
        if not self.profiles_data:
            return {"error": "No profiles loaded."}
        
        # Count skills and technologies mentioned
        skill_counts = {}
        for profile in self.profiles_data:
            profile_text = self._profile_to_text(profile).lower()
            
            # Count common skills
            skills = [
                'python', 'java', 'javascript', 'ai', 'ml', 'machine learning',
                'deep learning', 'data science', 'web development', 'cloud',
                'sql', 'react', 'angular', 'node.js', 'django', 'flask'
            ]
            
            for skill in skills:
                if skill in profile_text:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        summary = {
            "total_profiles": len(self.profiles_data),
            "top_skills": dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        return summary
    
    def scrape_linkedin_profiles(self, profile_urls: List[str]) -> Dict:
        """Scrape LinkedIn profiles and add to the database"""
        if not LINKEDIN_AVAILABLE:
            return {"success": False, "message": "LinkedIn scraping dependencies not available"}
        
        try:
            print(f"🔍 Starting to scrape {len(profile_urls)} LinkedIn profiles...")
            
            # Create a robust Chrome driver with multiple fallback options
            driver = self._create_robust_chrome_driver()
            if not driver:
                return {"success": False, "message": "Failed to create Chrome driver. Please check Chrome installation and try again."}
            
            # Try to login
            try:
                print("🔐 Attempting to login to LinkedIn...")
                actions.login(driver, self.linkedin_email, self.linkedin_password)
                print("✅ Login successful")
            except Exception as login_error:
                print(f"⚠️ Login failed: {login_error}")
                driver.quit()
                return {"success": False, "message": f"LinkedIn login failed: {str(login_error)}"}
            
            driver.quit()
            
            new_profiles = []
            successful_scrapes = 0
            failed_scrapes = 0
            
            for i, profile_url in enumerate(profile_urls, 1):
                try:
                    print(f"📊 Scraping Profile {i}/{len(profile_urls)}: {profile_url}")
                    
                    # Use a new driver for each profile
                    driver = self._create_robust_chrome_driver()
                    if not driver:
                        failed_scrapes += 1
                        print(f"  ⚠️ Failed to create driver for {profile_url}")
                        continue
                    
                    # Login for each profile
                    try:
                        actions.login(driver, self.linkedin_email, self.linkedin_password)
                    except Exception as login_error:
                        print(f"  ⚠️ Login failed for {profile_url}: {login_error}")
                        driver.quit()
                        failed_scrapes += 1
                        continue
                    person = Person(profile_url, driver=driver)
                    
                    # Process education data and remove duplicates
                    education_data = [{k: self._clean_text(v) for k, v in edu.__dict__.items()} for edu in person.educations]
                    education_data = self._remove_duplicate_education(education_data)
                    
                    data = {
                        "name": self._clean_text(person.name),
                        "about": self._clean_text(person.about),
                        "experiences": [{k: self._clean_text(v) for k, v in exp.__dict__.items()} for exp in person.experiences],
                        "education": education_data,
                        "linkedin_url": self._clean_text(person.linkedin_url)
                    }
                    new_profiles.append(data)
                    successful_scrapes += 1
                    print(f"  ✅ Successfully scraped: {data['name']}")
                    driver.quit()
                    
                except Exception as e:
                    failed_scrapes += 1
                    print(f"  ⚠️ Failed to scrape {profile_url}: {e}")
                    if 'driver' in locals():
                        driver.quit()
            
            # Load existing data and combine
            existing_data = self.profiles_data.copy()
            combined_data = existing_data + new_profiles
            
            # Remove duplicate profiles based on LinkedIn URL
            seen_urls = set()
            unique_data = []
            duplicates_removed = 0
            
            for profile in combined_data:
                linkedin_url = profile.get('linkedin_url', '')
                if linkedin_url and linkedin_url not in seen_urls:
                    seen_urls.add(linkedin_url)
                    unique_data.append(profile)
                else:
                    duplicates_removed += 1
            
            # Save combined results to JSON file
            with open(self.json_file_path, "w", encoding="utf-8") as f:
                json.dump(unique_data, f, indent=4, ensure_ascii=False)
            
            # Update the profiles data
            self.profiles_data = unique_data
            
            message = f"Successfully scraped {successful_scrapes} profiles, {failed_scrapes} failed. Total profiles: {len(unique_data)}"
            if duplicates_removed > 0:
                message += f" ({duplicates_removed} duplicates removed)"
            
            return {"success": True, "message": message}
            
        except Exception as e:
            error_msg = str(e)
            if "GetHandleVerifier" in error_msg or "Stacktrace" in error_msg:
                return {
                    "success": False, 
                    "message": "ChromeDriver error detected. Please ensure Chrome browser is installed and run: pip install webdriver-manager"
                }
            elif "chromedriver" in error_msg.lower():
                return {
                    "success": False, 
                    "message": "ChromeDriver not found. Please install webdriver-manager: pip install webdriver-manager"
                }
            else:
                return {"success": False, "message": f"Scraping failed: {error_msg}"}
    
    def _clean_text(self, text):
        """Clean text by removing newlines"""
        if isinstance(text, str):
            return text.replace("\n", " ").strip()
        return text
    
    def _remove_duplicate_education(self, education_list):
        """Remove duplicate education entries from a list"""
        if not education_list:
            return education_list
        
        seen_entries = set()
        unique_education = []
        duplicates_removed = 0
        
        for edu in education_list:
            # Create a hash of the education entry to identify duplicates
            edu_str = json.dumps(edu, sort_keys=True)
            edu_hash = hashlib.md5(edu_str.encode()).hexdigest()
            
            if edu_hash not in seen_entries:
                seen_entries.add(edu_hash)
                unique_education.append(edu)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            print(f"  🔧 Removed {duplicates_removed} duplicate education entry(ies)")
        
        return unique_education
    
    def _create_robust_chrome_driver(self):
        """Create a Chrome driver with multiple fallback options"""
        try:
            # Method 1: Try with webdriver-manager (most reliable)
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                
                print("🔧 Attempting to create Chrome driver with webdriver-manager...")
                service = Service(ChromeDriverManager().install())
                chrome_options = self._get_chrome_options()
                driver = webdriver.Chrome(service=service, options=chrome_options)
                print("✅ Chrome driver created successfully with webdriver-manager")
                return driver
            except Exception as e:
                print(f"⚠️ webdriver-manager failed: {e}")
            
            # Method 2: Try with system ChromeDriver
            try:
                print("🔧 Attempting to create Chrome driver with system ChromeDriver...")
                chrome_options = self._get_chrome_options()
                driver = webdriver.Chrome(options=chrome_options)
                print("✅ Chrome driver created successfully with system ChromeDriver")
                return driver
            except Exception as e:
                print(f"⚠️ System ChromeDriver failed: {e}")
            
            # Method 3: Try with specific ChromeDriver path (Windows)
            try:
                import os
                import platform
                
                if platform.system() == "Windows":
                    print("🔧 Attempting to create Chrome driver with Windows-specific path...")
                    chrome_options = self._get_chrome_options()
                    
                    # Common ChromeDriver paths on Windows
                    possible_paths = [
                        "chromedriver.exe",
                        "C:\\chromedriver.exe",
                        os.path.join(os.getcwd(), "chromedriver.exe"),
                        os.path.join(os.path.dirname(__file__), "chromedriver.exe")
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            service = Service(path)
                            driver = webdriver.Chrome(service=service, options=chrome_options)
                            print(f"✅ Chrome driver created successfully with path: {path}")
                            return driver
                    
                    print("⚠️ No ChromeDriver found in common Windows paths")
            except Exception as e:
                print(f"⚠️ Windows-specific ChromeDriver failed: {e}")
            
            # Method 4: Try with minimal options
            try:
                print("🔧 Attempting to create Chrome driver with minimal options...")
                chrome_options = Options()
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--disable-extensions")
                chrome_options.add_argument("--disable-logging")
                chrome_options.add_argument("--log-level=3")
                chrome_options.add_argument("--silent")
                chrome_options.add_argument("--disable-web-security")
                chrome_options.add_argument("--allow-running-insecure-content")
                chrome_options.add_argument("--disable-features=VizDisplayCompositor")
                
                driver = webdriver.Chrome(options=chrome_options)
                print("✅ Chrome driver created successfully with minimal options")
                return driver
            except Exception as e:
                print(f"⚠️ Minimal options ChromeDriver failed: {e}")
            
            print("❌ All Chrome driver creation methods failed")
            return None
            
        except Exception as e:
            print(f"❌ Error in _create_robust_chrome_driver: {e}")
            return None
    
    def _get_chrome_options(self):
        """Get optimized Chrome options for LinkedIn scraping"""
        chrome_options = Options()
        
        # Basic options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--silent")
        
        # Performance options
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
        chrome_options.add_argument("--disable-features=WebRtcAllowLegacyTLSProtocols")
        
        # Memory and stability options
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-field-trial-config")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        
        # User agent to avoid detection
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Experimental options for stability
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        return chrome_options

# Initialize the RAG app
rag_app = LinkedInRAGApp("linkedin_profiless_ls3.json")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/setup', methods=['POST'])
def setup_system():
    """Setup the RAG system"""
    try:
        # Setup vector store
        if not rag_app.setup_vectorstore():
            return jsonify({"success": False, "message": "Failed to setup vector store"})
        
        # Setup QA chain
        if not rag_app.setup_qa_chain():
            return jsonify({"success": False, "message": "Failed to setup QA chain"})
        
        return jsonify({"success": True, "message": "System setup completed successfully!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Setup failed: {str(e)}"})

@app.route('/api/query', methods=['POST'])
def ask_question():
    """Ask a question to the RAG system"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "message": "Please enter a question."})
        
        answer = rag_app.query(question)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing query: {str(e)}"})

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get database summary"""
    try:
        summary = rag_app.get_profile_summary()
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error getting summary: {str(e)}"})

@app.route('/api/scrape', methods=['POST'])
def scrape_profiles():
    """Scrape LinkedIn profiles"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({"success": False, "message": "No URLs provided"})
        
        if not LINKEDIN_AVAILABLE:
            return jsonify({"success": False, "message": "LinkedIn scraping dependencies not available. Please install linkedin-scraper and selenium."})
        
        result = rag_app.scrape_linkedin_profiles(urls)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error during scraping: {str(e)}"})



if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    create_html_template()
    
    print("🚀 Starting LinkedIn RAG Web Application...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
