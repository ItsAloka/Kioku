; Kioku per-user installer (no admin rights needed). Built by packaging\build.py:
;
;   .venv\Scripts\python.exe packaging\build.py
;
; Program files go to %LOCALAPPDATA%\Programs\Kioku. The photo index lives in
; %LOCALAPPDATA%\Kioku and is left alone on uninstall unless the user ticks the box to remove it.
; Kioku never changes photos; the uninstaller never touches them either.

#ifndef AppVersion
  #error Pass /DAppVersion=<version> (build.py does this)
#endif

#define AppName "Kioku"
#define AppExe "Kioku.exe"

[Setup]
AppId={{3C2E7A91-4B6D-4F0A-8E15-9D7C2B41A6F3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Aloka Warnakula
AppPublisherURL=https://github.com/ItsAloka/Kioku
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=Kioku-Setup-{#AppVersion}
SetupIconFile=..\src\kioku\resources\app.ico
UninstallDisplayIcon={app}\{#AppExe}
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\Kioku\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; AppUserModelID must match the one Kioku sets at startup (src\kioku\__main__.py), so the taskbar
; ties the running window to this shortcut and shows its icon (and pinning works).
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; AppUserModelID: "Kioku.PhotoSearch"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; AppUserModelID: "Kioku.PhotoSearch"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and
     (MsgBox('Also delete Kioku''s photo index (%LOCALAPPDATA%\Kioku)?' + #13#10 +
             'Your photos are not affected either way.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
    DelTree(ExpandConstant('{localappdata}\Kioku'), True, True, True);
end;
