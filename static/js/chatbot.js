/**
 * AI Traffic Assistant Chatbot
 * Real-time chat with Gemini API using traffic data context
 */

class TrafficChatbot {
    constructor() {
        this.messagesContainer = document.getElementById('chatbot-messages');
        this.inputField = document.getElementById('chatbot-input');
        this.sendBtn = document.getElementById('chatbot-send');
        this.loadingIndicator = document.getElementById('chatbot-loading');
        this.chatbotToggleBtn = document.getElementById('chatbot-toggle');
        this.chatbotContainer = document.getElementById('chatbot-container');
        
        this.isLoading = false;
        this.conversationHistory = [];
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.inputField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Chatbot toggle
        this.chatbotToggleBtn.addEventListener('click', () => this.toggleChatbot());
        
        // Auto-scroll to latest message
        this.messagesContainer.addEventListener('DOMNodeInserted', () => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        });
        
        console.log('Traffic Chatbot initialized');
    }
    
    toggleChatbot() {
        const isHidden = this.chatbotContainer.style.display === 'none';
        this.chatbotContainer.style.display = isHidden ? 'flex' : 'none';
        this.chatbotToggleBtn.textContent = isHidden ? '−' : '+';
    }
    
    sendMessage() {
        const message = this.inputField.value.trim();
        
        if (!message) return;
        
        if (this.isLoading) {
            this.showNotification('⏳ Please wait, the AI is thinking...', 'warning');
            return;
        }
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.inputField.value = '';
        
        // Send to server
        this.getAIResponse(message);
    }
    
    addMessage(text, sender = 'bot') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = this.escapeHtml(text);
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        // Store in history
        this.conversationHistory.push({
            sender: sender,
            text: text,
            timestamp: new Date()
        });
    }
    
    async getAIResponse(userMessage) {
        try {
            this.isLoading = true;
            this.showLoading(true);
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get response');
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessage(data.response, 'bot');
            } else {
                throw new Error(data.error || 'Unexpected error');
            }
        } catch (error) {
            console.error('Chat error:', error);
            const errorMsg = error.message || 'Error communicating with AI. Please try again.';
            this.addMessage(`❌ ${errorMsg}`, 'bot');
        } finally {
            this.isLoading = false;
            this.showLoading(false);
            this.inputField.focus();
        }
    }
    
    showLoading(show) {
        this.loadingIndicator.style.display = show ? 'flex' : 'none';
    }
    
    showNotification(text, type = 'info') {
        // Create temporary notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = text;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${type === 'warning' ? '#ff6f00' : '#43a047'};
            color: white;
            border-radius: 4px;
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    clearHistory() {
        this.conversationHistory = [];
        this.messagesContainer.innerHTML = `
            <div class="chat-message bot-message">
                <div class="message-content">
                    <p>👋 Hello! I'm your AI Traffic Assistant. Ask me anything about the current traffic analysis, signal timings, vehicle counts, or get recommendations for traffic management.</p>
                </div>
            </div>
        `;
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the results page
    if (document.getElementById('chatbot-messages')) {
        window.trafficChatbot = new TrafficChatbot();
    }
});
