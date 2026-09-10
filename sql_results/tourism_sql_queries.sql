
-- =========================================================
-- TOURISM EXPERIENCE ANALYTICS
-- SQL ANALYSIS QUERIES
-- =========================================================


-- 1. Rating Analysis by Visit Mode
SELECT
    VisitModeName AS Visit_Mode,
    COUNT(*) AS Total_Visits,
    ROUND(AVG(Rating), 2) AS Average_Rating
FROM tourism_data
GROUP BY VisitModeName
ORDER BY Average_Rating DESC;


-- 2. Top 10 Most Visited Attractions
SELECT
    Attraction,
    COUNT(*) AS Total_Visits,
    ROUND(AVG(Rating), 2) AS Average_Rating
FROM tourism_data
GROUP BY Attraction
ORDER BY Total_Visits DESC
LIMIT 10;


-- 3. Top 10 Cities by Tourism Visits
SELECT
    CityName AS City,
    COUNT(*) AS Total_Visits,
    ROUND(AVG(Rating), 2) AS Average_Rating,
    COUNT(DISTINCT UserId) AS Unique_Visitors
FROM tourism_data
WHERE CityName IS NOT NULL
GROUP BY CityName
ORDER BY Total_Visits DESC
LIMIT 10;


-- 4. Attraction Type Analysis
SELECT
    AttractionType,
    COUNT(*) AS Total_Visits,
    COUNT(DISTINCT AttractionId) AS Number_of_Attractions,
    ROUND(AVG(Rating), 2) AS Average_Rating
FROM tourism_data
WHERE AttractionType IS NOT NULL
GROUP BY AttractionType
ORDER BY Total_Visits DESC
LIMIT 10;


-- 5. Seasonal Tourism Analysis
SELECT
    VisitSeason AS Season,
    COUNT(*) AS Total_Visits,
    COUNT(DISTINCT UserId) AS Unique_Visitors,
    ROUND(AVG(Rating), 2) AS Average_Rating
FROM tourism_data
WHERE VisitSeason IS NOT NULL
GROUP BY VisitSeason
ORDER BY Total_Visits DESC;
