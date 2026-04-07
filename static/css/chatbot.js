class HealthChatbot {
    constructor() {
        // this.chatContainer = document.getElementById('chatMessages');
        this.chatContainer = document.getElementById('messageContainer');
        this.messageInput = document.getElementById('messageInput');
        this.fileInput = document.getElementById('fileInput');
        this.filePreview = document.getElementById('filePreview');
        this.previewContent = document.getElementById('previewContent');
        this.welcomeSection = document.getElementById('welcomeSection');
        this.selectedFile = null;
        this.isProcessing = false;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Enter key to send message
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Drag and drop for files
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        
        document.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect({ target: { files: e.dataTransfer.files } });
            }
        });
    }
    
    // async sendMessage() {
    //     const message = this.messageInput.value.trim();
        
    //     if ((!message && !this.selectedFile) || this.isProcessing) {
    //         return;
    //     }
        
    //     this.isProcessing = true;
        
    //     // Hide welcome section
    //     if (this.welcomeSection.style.display !== 'none') {
    //         this.welcomeSection.style.display = 'none';
    //     }
        
    //     // Add user message to chat
    //     this.addMessage(message, 'user', this.selectedFile);
        
    //     // Clear input and file
    //     this.messageInput.value = '';
    //     this.clearFile();
        
    //     // Show typing indicator
    //     const typingId = this.showTypingIndicator();
        
    //     try {
    //         const formData = new FormData();
    //         formData.append('message', message);
            
    //         if (this.selectedFile) {
    //             formData.append('file', this.selectedFile);
    //         }
            
    //         const response = await fetch('/chatbot', {
    //             method: 'POST',
    //             body: formData
    //         });
            
    //         const data = await response.json();
            
    //         // Remove typing indicator
    //         this.removeTypingIndicator(typingId);
            
    //         // Add bot response
    //         this.addMessage(data.response, 'bot', null, data.has_file);
            
    //         // Auto-scroll to bottom
    //         this.scrollToBottom();
            
    //     } catch (error) {
    //         this.removeTypingIndicator(typingId);
    //         this.addMessage('Sorry, I encountered an error. Please try again.', 'bot', null, false, true);
    //         console.error('Error:', error);
    //     } finally {
    //         this.isProcessing = false;
    //     }
    // }
    

    async sendMessage() {

    const message = this.messageInput.value.trim();
    const file = this.selectedFile;   // ⭐ save file first

    if ((!message && !file) || this.isProcessing) {
        return;
    }

    this.isProcessing = true;

    if (this.welcomeSection.style.display !== 'none') {
        this.welcomeSection.style.display = 'none';
    }

    // Add user message
    this.addMessage(message, 'user', file);

    // Clear input
    this.messageInput.value = '';
    this.clearFile();

    const typingId = this.showTypingIndicator();

    try {

        const formData = new FormData();
        formData.append('message', message);

        if (file) {
            formData.append('file', file);   // ⭐ use saved file
        }

        const response = await fetch('/chatbot', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        this.removeTypingIndicator(typingId);

        this.addMessage(data.response, 'bot', null, data.has_file);

        this.scrollToBottom();

    } catch (error) {

        this.removeTypingIndicator(typingId);

        this.addMessage(
            'Sorry, I encountered an error. Please try again.',
            'bot',
            null,
            false,
            true
        );

        console.error('Error:', error);

    } finally {
        this.isProcessing = false;
    }
}

 toggleFileUpload() {

    const uploadArea = document.getElementById("fileUploadArea");

    if (uploadArea.style.display === "none" || uploadArea.style.display === "") {
        uploadArea.style.display = "block";
    } else {
        uploadArea.style.display = "none";
    }

}

    addMessage(text, sender, file = null, hasFile = false, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message ${isError ? 'error-message' : ''}`;
        
        let content = `<div class="message-header">
            <span class="sender-icon">
                ${sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'}
            </span>
            <span class="sender-name">${sender === 'user' ? 'You' : 'Health Assistant'}</span>
            <span class="message-time">${this.getCurrentTime()}</span>
        </div>`;
        
        if (file) {
            content += `<div class="message-file">
                <i class="fas ${this.getFileIcon(file.type)}"></i>
                <span>${file.name}</span>
            </div>`;
        }
        
        if (text) {
            content += `<div class="message-text">${this.formatMessage(text)}</div>`;
        }
        
        if (hasFile && sender === 'bot') {
            content += `<div class="message-disclaimer">
                <i class="fas fa-microscope"></i>
                <span>Analysis provided for educational purposes only</span>
            </div>`;
        }
        
        messageDiv.innerHTML = content;
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(text) {
        // Format markdown-like syntax
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
            .replace(/\n/g, '<br>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^- (.*$)/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
    }
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }
        
        this.selectedFile = file;
        this.showFilePreview(file);
    }
    
    showFilePreview(file) {
        this.filePreview.style.display = 'block';
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.previewContent.innerHTML = `
                    <div class="image-preview">
                        <img src="${e.target.result}" alt="Preview">
                        <div class="image-info">
                            <i class="fas fa-image"></i>
                            <span>${file.name}</span>
                            <span class="file-size">(${this.formatFileSize(file.size)})</span>
                        </div>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        } else if (file.type === 'application/pdf') {
            this.previewContent.innerHTML = `
                <div class="pdf-preview">
                    <i class="fas fa-file-pdf pdf-icon"></i>
                    <div class="pdf-info">
                        <strong>${file.name}</strong>
                        <span class="file-size">${this.formatFileSize(file.size)}</span>
                        <span class="file-hint">Will analyze text content</span>
                    </div>
                </div>
            `;
        }
    }
    
    clearFile() {
        this.selectedFile = null;
        this.filePreview.style.display = 'none';
        this.previewContent.innerHTML = '';
        this.fileInput.value = '';
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typing-' + Date.now();
        typingDiv.innerHTML = `
            <div class="message-header">
                <span class="sender-icon"><i class="fas fa-robot"></i></span>
                <span class="sender-name">Health Assistant</span>
            </div>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.chatContainer.appendChild(typingDiv);
        this.scrollToBottom();
        return typingDiv.id;
    }
    
    removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    getFileIcon(fileType) {
        if (fileType.startsWith('image/')) return 'fa-image';
        if (fileType === 'application/pdf') return 'fa-file-pdf';
        if (fileType === 'text/plain') return 'fa-file-alt';
        return 'fa-file';
    }
    
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    clearChat() {
        if (confirm('Clear all chat messages?')) {
            this.chatContainer.innerHTML = '';
            this.welcomeSection.style.display = 'block';
        }
    }
    
    setQuickQuestion(question) {
        this.messageInput.value = question;
        this.messageInput.focus();
    }
    
    showFeatureInfo(feature) {
        const modal = document.getElementById('featureModal');
        const content = document.getElementById('modalContent');
        
        const featureInfo = {
            emergency: `
                <h3><i class="fas fa-ambulance"></i> Emergency Information</h3>
                <p><strong>This is not an emergency service.</strong></p>
                <p>If you're experiencing any of the following, seek immediate medical help:</p>
                <ul>
                    <li>Chest pain or pressure</li>
                    <li>Difficulty breathing</li>
                    <li>Severe bleeding</li>
                    <li>Sudden weakness or numbness</li>
                    <li>Thoughts of self-harm</li>
                </ul>
                <p><strong>Emergency Numbers:</strong></p>
                <ul>
                    <li>Emergency Services: 911 (US) or your local emergency number</li>
                    <li>Suicide Prevention: 988 (US)</li>
                    <li>Poison Control: 1-800-222-1222 (US)</li>
                </ul>
            `,
            privacy: `
                <h3><i class="fas fa-user-shield"></i> Privacy & Security</h3>
                <p><strong>How we protect your information:</strong></p>
                <ul>
                    <li>Files are processed temporarily and deleted immediately</li>
                    <li>Chat data is not stored permanently</li>
                    <li>No personal health information is collected</li>
                    <li>All communication is encrypted</li>
                </ul>
                <p><strong>Important:</strong> Do not share sensitive personal health information (PHI). 
                This assistant is for educational purposes only and is not HIPAA compliant for protected health information.</p>
                <p>For full privacy policy, consult with healthcare IT professionals.</p>
            `
        };
        
        content.innerHTML = featureInfo[feature] || '<p>Information not available.</p>';
        modal.style.display = 'block';
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new HealthChatbot();
});

// Global functions for HTML onclick handlers
function sendMessage() {
    if (window.chatbot) window.chatbot.sendMessage();
}

function handleKeyPress(event) {
    if (window.chatbot) window.chatbot.handleKeyPress(event);
}

function handleFileSelect(event) {
    if (window.chatbot) window.chatbot.handleFileSelect(event);
}

function clearFile() {
    if (window.chatbot) window.chatbot.clearFile();
}

function clearChat() {
    if (window.chatbot) window.chatbot.clearChat();
}

function setQuickQuestion(question) {
    if (window.chatbot) window.chatbot.setQuickQuestion(question);
}

function showFeatureInfo(feature) {
    if (window.chatbot) window.chatbot.showFeatureInfo(feature);
}

function closeModal() {
    document.getElementById('featureModal').style.display = 'none';
}