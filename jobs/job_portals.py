"""Module for handling job portal integrations"""
import urllib.parse
from typing import Dict, List

class JobPortal:
    """Class to handle job portal integrations and searches"""
    
    def __init__(self):
        """Initialize job portal connections"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.portals = [
            {
                "name": "LinkedIn",
                "icon": "fab fa-linkedin",
                "color": "#0077b5",
                "url": "https://www.linkedin.com/jobs/search/?keywords={}&location={}&f_E={}"
            },
            {
                "name": "Indeed",
                "icon": "fas fa-search-dollar",
                "color": "#2164f3",
                "url": "https://www.indeed.com/jobs?q={}&l={}&explvl={}"
            },
            {
                "name": "Naukri",
                "icon": "fas fa-briefcase",
                "color": "#4a90e2",
                "url": "https://www.naukri.com/{}-jobs-in-{}?experience={}"
            },
            {
                "name": "Foundit",
                "icon": "fas fa-globe",
                "color": "#ff6b6b",
                "url": "https://www.foundit.in/srp/results?query=\"{}\"&locations={}&experienceRanges={}~{}&experience={}"
            },
            {
                "name": "Instahyre",
                "icon": "fas fa-user-tie",
                "color": "#00bfa5",
                "url": "https://www.instahyre.com/{}-jobs-in-{}"
            },
            {
                "name": "Freshersworld",
                "icon": "fas fa-graduation-cap",
                "color": "#28a745",
                "url": "https://www.freshersworld.com/jobs/jobsearch/{}-jobs-in-{}"
            }
        ]

    def get_portal_list(self) -> List[Dict]:
        """Get list of available job portals"""
        return self.portals

    def format_query(self, query: str) -> str:
        """Format query string for URLs"""
        # Replace spaces with appropriate characters based on portal
        return query.replace(" ", "+")

    def format_location(self, location: str) -> str:
        """Format location string for URLs"""
        # Convert to lowercase and replace spaces with hyphens
        return location.strip().lower().replace(" ", "-")

    def format_job_title(self, title: str) -> str:
        """Format job title for URLs"""
        # Remove common words and special characters
        title = title.lower()
        title = title.replace("developer", "").replace("engineer", "").strip()
        title = title.replace(" ", "-")
        return title.strip("-")

    def format_experience(self, experience: str) -> tuple:
        """Format experience for different job portals"""
        if not experience or experience == "all":
            return "", "0", "0", "entry"
        
        try:
            # Handle dictionary input
            if isinstance(experience, dict):
                exp_id = experience.get('id', 'all')
                if exp_id == 'all':
                    return "", "0", "0", "entry"
                
                # Split experience range (e.g., "1-3" -> ["1", "3"])
                exp_min, exp_max = exp_id.split('-')
                if exp_max == "+":
                    exp_max = "15"  # Set a reasonable maximum for 10+ years
                
                # Map to portal-specific format
                exp_level = {
                    "0-1": "0",
                    "1-3": "1",
                    "3-5": "2",
                    "5-7": "3",
                    "7-10": "4",
                    "10+": "5"
                }.get(exp_id, "0")
                
                return exp_level, exp_min, exp_max, "entry" if exp_min == "0" else "experienced"
            
            return "", "0", "0", "entry"
            
        except Exception as e:
            print(f"Error formatting experience: {str(e)}")
            return "", "0", "0", "entry"

    def search_jobs(self, query: str, location: str = "", experience: dict = None) -> list:
        """Search for jobs across all portals with formatted URLs"""
        results = []
        formatted_query = self.format_query(query)
        formatted_location = self.format_location(location) if location else ""
        job_title = self.format_job_title(query)
        
        # Handle experience
        if experience and isinstance(experience, dict):
            exp_id = experience.get('id', 'all')
            exp_text = experience.get('text', 'All Levels')
            exp_level, exp_min, exp_max, exp_type = self.format_experience(exp_id)
        else:
            exp_level, exp_min, exp_max, exp_type = "", "0", "0", "entry"
        
        for portal in self.portals:
            try:
                if portal["name"] == "Foundit":
                    url = portal["url"].format(
                        formatted_query,
                        formatted_location,
                        exp_min, exp_max, exp_min
                    )
                elif portal["name"] == "Instahyre":
                    url = portal["url"].format(
                        job_title,  # Use simplified job title
                        formatted_location
                    )
                elif portal["name"] == "Freshersworld":
                    url = portal["url"].format(
                        job_title,  # Use simplified job title
                        formatted_location
                    )
                elif portal["name"] in ["LinkedIn", "Indeed", "Naukri"]:
                    url = portal["url"].format(
                        formatted_query,
                        formatted_location,
                        exp_level
                    )
                else:
                    url = portal["url"].format(formatted_query, formatted_location)
                
                results.append({
                    "portal": portal["name"],
                    "icon": portal["icon"],
                    "color": portal["color"],
                    "title": f"Search {query} jobs in {location}" if location else f"Search {query} jobs",
                    "url": url
                })
            except Exception as e:
                print(f"Error generating URL for {portal['name']}: {str(e)}")
                continue
        
        return results
