# SQLite Installation Guide / SQLite Installations-Anleitung

This guide explains how to install and use SQLite for the Flight Delay Detection project.

*Diese Anleitung erklärt, wie SQLite für das Flugverspätungs-Erkennungsprojekt installiert und verwendet wird.*

## 📋 Table of Contents / Inhaltsverzeichnis

- [What is SQLite? / Was ist SQLite?](#what-is-sqlite--was-ist-sqlite)
- [Installation / Installation](#installation--installation)
- [Key Features / Wichtige Eigenschaften](#key-features--wichtige-eigenschaften)
- [Usage in This Project / Verwendung in diesem Projekt](#usage-in-this-project--verwendung-in-diesem-projekt)
- [SQLite Commands / SQLite-Befehle](#sqlite-commands--sqlite-befehle)
- [Special Considerations / Besonderheiten](#special-considerations--besonderheiten)

## What is SQLite? / Was ist SQLite?

SQLite is a lightweight, serverless, self-contained SQL database engine. It stores the entire database in a single file, making it ideal for local applications and development.

*SQLite ist eine leichtgewichtige, serverlose, eigenständige SQL-Datenbank-Engine. Sie speichert die gesamte Datenbank in einer einzigen Datei, was sie ideal für lokale Anwendungen und Entwicklung macht.*

### Key Advantages / Hauptvorteile

- **Zero Configuration**: No server setup required / *Keine Konfiguration: Kein Server-Setup erforderlich*
- **Portable**: Single file database / *Portabel: Einzeldatei-Datenbank*
- **Lightweight**: Minimal resource usage / *Leichtgewichtig: Minimaler Ressourcenverbrauch*
- **ACID Compliant**: Full transaction support / *ACID-konform: Vollständige Transaktionsunterstützung*

## Installation / Installation

### Python Integration / Python-Integration

SQLite is **already included** in Python's standard library (since Python 2.5). No additional installation is required!

*SQLite ist **bereits enthalten** in der Python-Standardbibliothek (seit Python 2.5). Keine zusätzliche Installation erforderlich!*

```python
import sqlite3

# Create/connect to database
conn = sqlite3.connect('flights2024.db')
```

### Using SQLAlchemy (Recommended for This Project) / Verwendung von SQLAlchemy (Empfohlen für dieses Projekt)

For the Flight Delay Detection project, we use **SQLAlchemy** for database operations:

*Für das Flugverspätungs-Erkennungsprojekt verwenden wir **SQLAlchemy** für Datenbankoperationen:*

```bash
pip install sqlalchemy
```

Or install all dependencies from requirements.txt:

*Oder installieren Sie alle Abhängigkeiten aus requirements.txt:*

```bash
pip install -r requirements.txt
```

### SQLite Command-Line Tool / SQLite Kommandozeilen-Tool

#### Windows

SQLite is not pre-installed on Windows. Download it from the official website:

*SQLite ist nicht vorinstalliert auf Windows. Laden Sie es von der offiziellen Website herunter:*

1. Visit / *Besuchen Sie*: https://www.sqlite.org/download.html
2. Download **sqlite-tools-win32-x86-*.zip**
3. Extract to a folder (e.g., `C:\sqlite`)
4. Add to PATH or run from the folder / *Zum PATH hinzufügen oder aus dem Ordner ausführen*

```powershell
# Run SQLite
sqlite3.exe flights2024.db
```

#### macOS

SQLite comes pre-installed on macOS.

*SQLite ist vorinstalliert auf macOS.*

```bash
# Verify installation
sqlite3 --version

# Open database
sqlite3 flights2024.db
```

#### Linux (Ubuntu/Debian)

SQLite is usually pre-installed. If not, install via package manager:

*SQLite ist normalerweise vorinstalliert. Falls nicht, Installation über den Paketmanager:*

```bash
# Install if needed
sudo apt-get update
sudo apt-get install sqlite3

# Verify installation
sqlite3 --version

# Open database
sqlite3 flights2024.db
```

## Key Features / Wichtige Eigenschaften

### File-Based Storage / Dateibasierte Speicherung

The entire database is stored in a single `.db` file:

*Die gesamte Datenbank wird in einer einzigen `.db`-Datei gespeichert:*

- `flights2024.db` - Contains all flight predictions / *Enthält alle Flugvorhersagen*
- Portable across platforms / *Portabel über Plattformen hinweg*
- Easy to backup and share / *Einfach zu sichern und zu teilen*

### Performance Considerations / Performance-Überlegungen

SQLite is optimized for:

*SQLite ist optimiert für:*

- **Read Operations**: Excellent for analytics / *Lese-Operationen: Hervorragend für Analysen*
- **Small to Medium Datasets**: Works well up to several GB / *Kleine bis mittlere Datensätze: Funktioniert gut bis zu mehreren GB*
- **Single User**: Best for local applications / *Einzelbenutzer: Am besten für lokale Anwendungen*

### Limitations / Einschränkungen

- No concurrent writes / *Keine gleichzeitigen Schreibvorgänge*
- Limited to local file system / *Begrenzt auf lokales Dateisystem*
- Not suitable for high-concurrency web apps / *Nicht geeignet für Web-Apps mit hoher Gleichzeitigkeit*

## Usage in This Project / Verwendung in diesem Projekt

### Database Structure / Datenbankstruktur

The project creates the following database and table:

*Das Projekt erstellt folgende Datenbank und Tabelle:*

**Database File / Datenbankdatei**: `flights2024.db`

**Table / Tabelle**: `flight_preds_2024`

Example table schema:

*Beispiel-Tabellenschema:*

```sql
CREATE TABLE flight_preds_2024 (
    id INTEGER PRIMARY KEY,
    fl_date DATE,
    op_unique_carrier TEXT,
    origin TEXT,
    dest TEXT,
    dep_delay REAL,
    arr_delay REAL,
    Delayed INTEGER,
    prediction INTEGER,
    -- ... additional columns
);
```

### Creating the Database / Datenbank erstellen

The database is automatically created when running the Jupyter notebook:

*Die Datenbank wird automatisch erstellt beim Ausführen des Jupyter Notebooks:*

```python
from sqlalchemy import create_engine

# Create SQLite engine
engine = create_engine('sqlite:///flights2024.db', echo=False)

# Store DataFrame to database
df.to_sql('flight_preds_2024', con=engine, if_exists='replace', index=False)
```

### Querying the Database / Datenbank abfragen

You can query the database using:

*Sie können die Datenbank abfragen mit:*

1. **Python (SQLAlchemy)**:
   ```python
   import pandas as pd
   from sqlalchemy import create_engine
   
   engine = create_engine('sqlite:///flights2024.db')
   df = pd.read_sql('SELECT * FROM flight_preds_2024 LIMIT 100', con=engine)
   ```

2. **SQLite Command Line**:
   ```bash
   sqlite3 flights2024.db
   SELECT * FROM flight_preds_2024 LIMIT 10;
   ```

## SQLite Commands / SQLite-Befehle

### Basic Commands / Grundlegende Befehle

Once in the SQLite shell (`sqlite3 flights2024.db`):

*Im SQLite-Shell (`sqlite3 flights2024.db`):*

```sql
-- List all tables / Alle Tabellen auflisten
.tables

-- Show table schema / Tabellenschema anzeigen
.schema flight_preds_2024

-- Execute SQL query / SQL-Abfrage ausführen
SELECT COUNT(*) FROM flight_preds_2024;

-- Enable column headers / Spaltenüberschriften aktivieren
.headers on

-- Set output mode / Ausgabemodus setzen
.mode column

-- Export to CSV / Export nach CSV
.mode csv
.output flight_predictions.csv
SELECT * FROM flight_preds_2024;
.output stdout

-- Exit SQLite / SQLite beenden
.quit
```

## Special Considerations / Besonderheiten

### 1. Transaction Management / Transaktionsverwaltung

SQLite uses file-level locking:

*SQLite verwendet Dateisperren:*

- Only one write transaction at a time / *Nur eine Schreibtransaktion gleichzeitig*
- Multiple read transactions are allowed / *Mehrere Lesetransaktionen sind erlaubt*

### 2. Data Types / Datentypen

SQLite has dynamic typing with five storage classes:

*SQLite hat dynamische Typisierung mit fünf Speicherklassen:*

- `NULL` - Null value / *Null-Wert*
- `INTEGER` - Signed integer / *Vorzeichenbehaftete Ganzzahl*
- `REAL` - Floating point / *Gleitkommazahl*
- `TEXT` - Text string / *Textzeichenkette*
- `BLOB` - Binary data / *Binärdaten*

### 3. File Size Optimization / Dateigrößen-Optimierung

For large datasets (like our 7M flight records):

*Für große Datensätze (wie unsere 7M Flugdatensätze):*

```sql
-- Optimize database file / Datenbankdatei optimieren
VACUUM;

-- Analyze for query optimization / Analyse für Query-Optimierung
ANALYZE;
```

### 4. Backup Strategy / Backup-Strategie

Simple file copy is sufficient:

*Einfaches Datei-Kopieren ist ausreichend:*

```bash
# Linux/macOS
cp flights2024.db flights2024_backup.db

# Windows (PowerShell)
Copy-Item flights2024.db flights2024_backup.db
```

### 5. Integrity Checks / Integritätsprüfungen

Verify database integrity:

*Datenbank-Integrität überprüfen:*

```bash
sqlite3 flights2024.db "PRAGMA integrity_check;"
```

## 🔗 Additional Resources / Zusätzliche Ressourcen

- **Official Documentation**: https://www.sqlite.org/docs.html
- **SQLAlchemy Documentation**: https://docs.sqlalchemy.org/
- **SQLite Tutorial**: https://www.sqlitetutorial.net/

## ⚠️ Important Notes / Wichtige Hinweise

1. **Don't commit large .db files to Git** / *Große .db-Dateien nicht zu Git committen*
   - Add to `.gitignore` / *Zur `.gitignore` hinzufügen*
   - Use DVC for data version control / *DVC für Daten-Versionskontrolle verwenden*

2. **Regular backups** for production data / *Regelmäßige Backups für Produktionsdaten*

3. **Use indexes** for frequently queried columns / *Indizes verwenden für häufig abgefragte Spalten*

```sql
CREATE INDEX idx_fl_date ON flight_preds_2024(fl_date);
CREATE INDEX idx_carrier ON flight_preds_2024(op_unique_carrier);
```

---

> ➡️ **Back to Main Documentation:** [README.md](../README.md)
