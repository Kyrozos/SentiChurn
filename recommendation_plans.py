# Define recommendation plans based on product score
recommendation_plans = {
                        "LOW" : {
                                "Features" : """Prioritize fixing critical bugs or usability issues identified in negative feedback.
Conduct user research to understand missing features causing frustration.""",
                                "UX" : """Analyze user recordings or heatmaps to identify confusing or clunky user flows.
Simplify the user interface based on user feedback.""",
                                "CS" : """Review common customer support tickets to identify areas where response time or resolution needs improvement.
Implement live chat or self-service options for faster resolution.""",
                                "Pricing" : """Analyze customer churn related to pricing and consider offering more flexible pricing plans.
Review competitor pricing strategies.""",
                                "Marketing" : """Re-evaluate marketing messaging based on negative sentiment to ensure it accurately reflects the product's value proposition.
Consider targeting a different customer segment if the current marketing isn't reaching the right audience."""
},
                        "MEDIUM" : {
                                "Features" : """Implement frequently requested features with high potential impact based on user feedback.
A/B test different feature variations to see what resonates best with users.""",
                                "UX" : """Conduct user interviews or surveys to gather detailed feedback on specific aspects of the user experience.
Prioritize UX improvements based on the severity of user pain points.""",
                                "CS" : """Invest in training customer support representatives to handle complex issues more effectively.
Implement customer satisfaction surveys to track improvement in support interactions.""",
                                "Pricing" : """Offer limited-time discounts or promotions to incentivize user retention.
Consider offering tiered pricing plans with different feature sets.""",
                                "Marketing" : """Analyze customer acquisition data to identify the most effective marketing channels.
Refine your marketing campaigns to target existing user segments more effectively."""
},
                        "HIGH" : {
                                "Features" : """Focus on adding innovative features that enhance the core user experience.
Conduct user research to identify potential new features that address unmet user needs.""",
                                "UX" : """Conduct user research to identify opportunities for further UX optimization and user delight.
A/B test different UI/UX variations to optimize user engagement.""",
                                "CS" : """Implement proactive outreach programs to high-value customers.
Offer self-service knowledge base articles and tutorials.""",
                                "Pricing" : """Consider offering loyalty programs or reward systems to incentivize long-term users.
Monitor competitor pricing strategies and adjust accordingly.""",
                                "Marketing" : """Develop customer referral programs to encourage user acquisition through existing satisfied customers.
Invest in building a strong brand community to foster user loyalty."""
}
}