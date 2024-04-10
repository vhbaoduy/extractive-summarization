// content.js

// Function to get text content of the current page
function getTextContent() {
    return document.body.textContent.trim();
}

// Get text content of the current page
const pageContent = getTextContent();

// Send the page content to the background script
chrome.runtime.sendMessage({ content: pageContent });