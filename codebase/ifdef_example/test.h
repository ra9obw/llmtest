#define GEN_DOC

/**
 * Attribute configuration data extension
 *
 * @headerfile tango.h
 * @ingroup Client
 */
// typedef and structure
#ifdef GEN_DOC
typedef struct AttributeInfo : public DeviceAttributeConfig
#else
typedef struct _AttributeInfo : public DeviceAttributeConfig
#endif
{
    int disp_level;
    bool tmp;
}AttributeInfo;