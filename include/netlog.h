#ifndef NETLOG_H
#define NETLOG_H

#define NETLOG_IP "192.168.1.101"
#define NETLOG_PORT 3490

void netlog_init(void);
void netlog_fini(void);
void netlog(const char *format, ...);

#endif
