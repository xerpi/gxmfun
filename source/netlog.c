#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <psp2/sysmodule.h>
#include <psp2/net/net.h>
#include "netlog.h"

#define NET_INIT_SIZE (64 * 1024)

static void *net_memory = NULL;
static SceNetSockaddrIn host_addr;
static int udp_sock;

void netlog_init(void)
{
	if (sceSysmoduleIsLoaded(SCE_SYSMODULE_NET) != SCE_SYSMODULE_LOADED)
		sceSysmoduleLoadModule(SCE_SYSMODULE_NET);

	if (sceNetShowNetstat() == SCE_NET_ERROR_ENOTINIT) {
		SceNetInitParam init_param;

		net_memory = malloc(NET_INIT_SIZE);
		if (!net_memory)
			return;

		init_param.memory = net_memory;
		init_param.size = NET_INIT_SIZE;
		init_param.flags = 0;
		sceNetInit(&init_param);
	}

	udp_sock = sceNetSocket("netlog", SCE_NET_AF_INET, SCE_NET_SOCK_DGRAM, 0);

	sceNetInetPton(SCE_NET_AF_INET, NETLOG_IP, &host_addr.sin_addr);
	host_addr.sin_family = SCE_NET_AF_INET;
	host_addr.sin_port = sceNetHtons(NETLOG_PORT);
}

void netlog_fini(void)
{
	sceNetSocketClose(udp_sock);

	sceNetTerm();

	if (sceSysmoduleIsLoaded(SCE_SYSMODULE_NET) == SCE_SYSMODULE_LOADED)
		sceSysmoduleUnloadModule(SCE_SYSMODULE_NET);

	if (net_memory) {
		free(net_memory);
		net_memory = NULL;
	}
}

static inline void udp_send(const char *buffer)
{
	sceNetSendto(udp_sock, buffer, strlen(buffer), SCE_NET_MSG_DONTWAIT,
		(SceNetSockaddr *)&host_addr, sizeof(host_addr));
}

void netlog(const char *format, ...)
{
	char buffer[512];
	va_list args;

	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer), format, args);
	udp_send(buffer);
	va_end(args);
}
