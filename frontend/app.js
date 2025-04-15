async function ask() {
    const questionBox = document.getElementById('question');
    const chatBox = document.getElementById('chat-box');
    const question = questionBox.value;

    if (!question.trim()) return;

    const userMsg = document.createElement('div');
    userMsg.className = 'user-msg';
    userMsg.innerText = question;
    chatBox.appendChild(userMsg);

    const res = await fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });

    const data = await res.json();
    data.answers.forEach(answer => {
        const botMsg = document.createElement('div');
        botMsg.className = 'bot-msg';
        botMsg.innerText = answer;
        chatBox.appendChild(botMsg);
    });

    questionBox.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
}
