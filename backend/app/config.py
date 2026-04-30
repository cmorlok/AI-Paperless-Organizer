from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/organizer.db"
    
    # App settings
    app_name: str = "AI Paperless Organizer"
    debug: bool = False


settings = Settings()

