export function prettyLog(title: string, data: any) {
  console.log(
    "%c==============================",
    "color:#6FFFE9; font-weight:bold;"
  );
  console.log(`%c${title}`, "color:#F75C7E; font-weight:bold; font-size:14px;");
  console.log(
    "%c==============================",
    "color:#6FFFE9; font-weight:bold;"
  );

  console.log(
    "%c" + JSON.stringify(data, null, 2),
    "color:#D1E8FF; font-family:monospace;"
  );

  console.log(
    "%c------------------------------\n\n",
    "color:#6FFFE9; font-weight:bold;"
  );
}
