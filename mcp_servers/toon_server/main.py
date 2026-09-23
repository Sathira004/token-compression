from fastmcp import FastMCP
import requests
import sys

mcp = FastMCP("ToonServer")


@mcp.tool()
def mcp_get_users() -> list:
    """
    Fetch users from the JSONPlaceholder API.

    Returns:
        A list of user records containing name, username,
        email, address, company, and other structured data.
    """
    url = "https://jsonplaceholder.typicode.com/users"

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    print(
        f"Fetched {len(response.json())} users from JSONPlaceholder API.",
        file=sys.stderr,
        flush=True,
    )

    return response.json()


@mcp.tool()
def mcp_get_posts(user_id: int | None = None) -> list:
    """
    Fetch posts from the JSONPlaceholder API.

    Args:
        user_id: Optional user ID to filter posts.
                 If omitted, all posts are returned.

    Returns:
        A list of post records containing userId, id,
        title, and body.
    """
    url = "https://jsonplaceholder.typicode.com/posts"

    params = {}

    if user_id is not None:
        params["userId"] = user_id

    response = requests.get(
        url,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    
    print(
        f"Fetched {len(response.json())} posts from JSONPlaceholder API.",
        file=sys.stderr,
        flush=True,
    )

    return response.json()


@mcp.tool()
def mcp_get_weather(
    latitude: float,
    longitude: float
) -> dict:
    """
    Fetch current weather data from Open-Meteo.

    Args:
        latitude: Latitude of the requested location.
        longitude: Longitude of the requested location.

    Returns:
        Structured weather data including temperature,
        wind speed, and weather information.
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    
    print(
        f"Fetched weather data for coordinates ({latitude}, {longitude}) from Open-Meteo API.",
        file=sys.stderr,
        flush=True,
    )

    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")