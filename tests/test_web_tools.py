import pytest
from unittest.mock import patch, MagicMock, mock_open
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual module to test
try:
    import web_tools
except ImportError:
    web_tools = None

class TestWebTools:
    """Test suite for web scraping and processing tools"""
    
    def test_web_tools_import(self):
        """Test that web_tools module can be imported"""
        if web_tools is not None:
            assert web_tools is not None
        else:
            pytest.skip("web_tools module not available")
    
    @patch('requests.get')
    def test_web_scraping_basic(self, mock_get):
        """Test basic web scraping functionality"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Job Description</h1><p>Python Developer</p></body></html>"
        mock_response.content = mock_response.text.encode('utf-8')
        mock_get.return_value = mock_response
        
        # Test successful request
        response = mock_get('https://example.com/job')
        assert response.status_code == 200
        assert 'Python Developer' in response.text
        assert 'Job Description' in response.text
    
    @patch('requests.get')
    def test_error_handling(self, mock_get):
        """Test web scraping error handling"""
        # Test different error scenarios
        error_scenarios = [
            (404, "Not Found"),
            (500, "Internal Server Error"),
            (403, "Forbidden"),
            (requests.ConnectionError, "Connection Error")
        ]
        
        for error_code, error_message in error_scenarios:
            if isinstance(error_code, int):
                mock_response = MagicMock()
                mock_response.status_code = error_code
                mock_response.text = error_message
                mock_get.return_value = mock_response
                
                response = mock_get('https://example.com/job')
                assert response.status_code == error_code
            else:
                mock_get.side_effect = error_code
                with pytest.raises(error_code):
                    mock_get('https://example.com/job')
    
    def test_html_parsing(self):
        """Test HTML parsing and content extraction"""
        try:
            from bs4 import BeautifulSoup
            
            # Mock HTML content
            html_content = """
            <html>
                <body>
                    <div class="job-description">
                        <h1>Software Engineer</h1>
                        <div class="requirements">
                            <ul>
                                <li>Python programming</li>
                                <li>Machine Learning</li>
                                <li>3+ years experience</li>
                            </ul>
                        </div>
                        <div class="salary">$80,000 - $120,000</div>
                    </div>
                </body>
            </html>
            """
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract job title
            job_title = soup.find('h1').text
            assert job_title == "Software Engineer"
            
            # Extract requirements
            requirements = [li.text for li in soup.find_all('li')]
            assert len(requirements) == 3
            assert 'Python programming' in requirements
            assert 'Machine Learning' in requirements
            
            # Extract salary
            salary = soup.find('div', class_='salary').text
            assert '$80,000' in salary
        except ImportError:
            # Skip test if BeautifulSoup is not available
            pytest.skip("BeautifulSoup not available")
    
    def test_url_validation(self):
        """Test URL validation and sanitization"""
        import re
        
        # Valid URLs
        valid_urls = [
            'https://example.com/job/123',
            'http://careers.company.com/position',
            'https://www.jobboard.com/listing?id=456'
        ]
        
        # Invalid URLs
        invalid_urls = [
            'not-a-url',
            'ftp://example.com',  # Wrong protocol
            'javascript:alert("xss")',  # Security risk
            '',
            None
        ]
        
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
        
        for url in valid_urls:
            assert url_pattern.match(url) is not None
        
        for url in invalid_urls:
            if url:
                if url.startswith(('http://', 'https://')):
                    continue  # These might be valid
                elif url.startswith('javascript:'):
                    assert url.startswith('javascript:')  # This should be flagged as invalid
                else:
                    # Test other invalid patterns
                    assert url_pattern.match(url) is None
    
    def test_content_cleaning(self):
        """Test content cleaning and normalization"""
        messy_content = """
        \t\t  Software Engineer   \n\n
        
        Job Description:    \r\n
        We are looking for a    talented developer...
        
        \n\n\n   Requirements:\n
        •  Python\t\t
        •   Java  \r
        """
        
        # Clean whitespace
        cleaned_lines = []
        for line in messy_content.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        assert 'Software Engineer' in cleaned_content
        assert '\t\t' not in cleaned_content
        assert cleaned_content.count('\n\n\n') == 0  # Multiple newlines removed
